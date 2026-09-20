// Loads a synthetic qwen4exp package at production dimensions and reports
// what the weights actually cost.
//
// The files are sparse: only each 16-byte header is written and the rest is a
// hole, so a package whose apparent size is tens of gibibytes occupies a few
// kibibytes on disk. That is what makes it possible to exercise the real
// 48-layer, 512-expert geometry without the real checkpoint.

#include "engine/MemoryPlan.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/Qwen4Exp.hpp"

#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
using namespace splash;
using namespace splash::model;
using splash::metal::MetalBackend;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

uint64_t alignPacked(uint64_t value) {
  return (value + kWeightFileAlignment - 1) & ~(kWeightFileAlignment - 1);
}
uint64_t q4Bytes(uint64_t outputSize, uint64_t inputSize) {
  return outputSize * inputSize * 9 / 16;
}

void storeLittleEndian32(uint8_t *destination, uint32_t value) {
  for (int index = 0; index < 4; ++index)
    destination[index] = uint8_t((value >> (8 * index)) & 0xFF);
}

// Header, then a hole. ftruncate gives the file its full apparent size.
uint64_t writeWeightFile(const std::filesystem::path &path,
                         std::string_view magic, uint32_t layer, uint32_t type,
                         std::span<const uint64_t> sections) {
  std::filesystem::create_directories(path.parent_path());
  int descriptor =
      open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  require(descriptor >= 0, "unable to create " + path.string());
  std::array<uint8_t, 16> header{};
  std::memcpy(header.data(), magic.data(), magic.size());
  storeLittleEndian32(header.data() + 8, layer);
  storeLittleEndian32(header.data() + 12, type);
  require(pwrite(descriptor, header.data(), header.size(), 0) == 16,
          "unable to write header");
  uint64_t offset = header.size();
  for (uint64_t bytes : sections) offset = alignPacked(offset) + bytes;
  uint64_t fileBytes = alignPacked(offset);
  require(ftruncate(descriptor, off_t(fileBytes)) == 0, "unable to size file");
  close(descriptor);
  return fileBytes;
}

// Mirrors loadQwen4ExpWeights section for section.
std::vector<uint64_t> hyperConnectionSections(const Qwen4ExpLayout &l) {
  const uint64_t width = uint64_t(l.hyperConnectionWidth()) * kBFloat16Bytes;
  return {width, l.hyperConnectionCount * width,
          l.hyperConnectionLowRank * width, l.hyperConnectionLowRank * width};
}

std::vector<uint64_t> layerSections(const Qwen4ExpLayout &l, bool full) {
  std::vector<uint64_t> s = hyperConnectionSections(l);
  if (full) {
    s.push_back(q4Bytes(l.packedFullWidth, l.hiddenSize));
    s.push_back(uint64_t(l.attentionHeadDimension) * kBFloat16Bytes);
    s.push_back(uint64_t(l.attentionHeadDimension) * kBFloat16Bytes);
    s.push_back(q4Bytes(l.hiddenSize, l.attentionWidth));
    s.push_back(q4Bytes(l.indexerProjectionWidth(), l.hiddenSize));
    s.push_back(uint64_t(l.indexerHeadDimension) * kBFloat16Bytes);
    s.push_back(uint64_t(l.indexerHeadDimension) * kBFloat16Bytes);
  } else {
    s.push_back(q4Bytes(l.packedGdnWidth, l.hiddenSize));
    s.push_back(uint64_t(l.convolutionDimension) * kGdnConvolutionTaps *
                kBFloat16Bytes);
    s.push_back(uint64_t(l.gdnValueHeads) * 4);
    s.push_back(uint64_t(l.gdnValueHeads) * kBFloat16Bytes);
    s.push_back(uint64_t(l.gdnHeadDimension) * kBFloat16Bytes);
    s.push_back(q4Bytes(l.hiddenSize, l.attentionWidth));
  }
  for (uint64_t bytes : hyperConnectionSections(l)) s.push_back(bytes);
  const uint64_t q8Router =
      uint64_t(l.experts) * l.hiddenSize +
      2 * (uint64_t(l.experts) * l.hiddenSize / 32);
  s.push_back(q8Router);
  s.push_back(l.experts * q4Bytes(l.expertIntermediateSize, l.hiddenSize));
  s.push_back(l.experts * q4Bytes(l.expertIntermediateSize, l.hiddenSize));
  s.push_back(l.experts * q4Bytes(l.hiddenSize, l.expertIntermediateSize));
  s.push_back(q4Bytes(l.expertIntermediateSize, l.hiddenSize));
  s.push_back(q4Bytes(l.expertIntermediateSize, l.hiddenSize));
  s.push_back(q4Bytes(l.hiddenSize, l.expertIntermediateSize));
  const uint64_t q8Gate = uint64_t(kQ4StorageN) * l.hiddenSize +
                          2 * (uint64_t(kQ4StorageN) * l.hiddenSize / 32);
  s.push_back(q8Gate);
  return s;
}

std::vector<uint64_t> draftLayerSections(const DFlashDraftLayout &d) {
  const uint64_t hidden = uint64_t(d.hiddenSize) * kBFloat16Bytes;
  const uint64_t conv = uint64_t(4) * d.hiddenSize * kBFloat16Bytes;
  const uint64_t headNorm =
      uint64_t(d.attentionHeadDimension) * kBFloat16Bytes;
  return {hidden,
          conv,
          q4Bytes(d.dynamicSize, d.hiddenSize),
          q4Bytes(d.qkvSize, d.hiddenSize),
          headNorm,
          headNorm,
          q4Bytes(d.hiddenSize, d.attentionSize),
          hidden,
          conv,
          q4Bytes(d.dynamicSize, d.hiddenSize),
          q4Bytes(d.intermediateSize, d.hiddenSize),
          q4Bytes(d.intermediateSize, d.hiddenSize),
          q4Bytes(d.hiddenSize, d.intermediateSize)};
}

std::string gib(uint64_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << double(bytes) / double(1ULL << 30) << " GiB";
  return out.str();
}

void writeManifest(const std::filesystem::path &root, const Qwen4ExpLayout &l,
                   const DFlashDraftLayout &d) {
  std::ostringstream m;
  m << "{\"model\":\"Qwen3.8-Flash-Next\",\"schema_version\":5,"
    << "\"format\":{\"name\":\"splash-packed-q4-qwen4exp\",\"q4_bits\":4,"
    << "\"q8_bits\":8,\"quant_group_size\":" << kQ4GroupElements
    << ",\"storage_n\":" << kQ4StorageN
    << ",\"expert_storage_n\":" << kQ4ExpertStorageN
    << ",\"section_alignment_bytes\":" << kWeightFileAlignment
    << ",\"target_layer_magic\":\"" << Qwen4ExpLayout::layerMagic
    << "\",\"draft_layer_magic\":\"MDFD0004\","
    << "\"vision_magic\":\"MDFV0001\"},"
    << "\"execution_geometry\":{"
    << "\"allocation_extent_target_bytes\":" << kv::kAllocationExtentTargetBytes
    << ",\"draft_proposal_tokens\":" << ExecutionLimits::draftProposalTokens
    << ",\"draft_query_rows\":" << ExecutionLimits::draftQueryRows
    << ",\"draft_sliding_window\":" << ExecutionLimits::draftContextTokens
    << ",\"maximum_batch_width\":" << ExecutionLimits::maximumBatchWidth
    << ",\"prefill_token_budget\":" << ExecutionLimits::prefillTokenBudget
    << ",\"target_kv_block_tokens\":" << kv::kPageTokens
    << ",\"target_verify_rows\":" << ExecutionLimits::targetVerifyRows << "},"
    << "\"target\":{\"architecture\":\"qwen4exp\",\"layers\":" << l.layers
    << ",\"hidden_size\":" << l.hiddenSize
    << ",\"vocabulary_size\":" << l.vocabularySize
    << ",\"gdn_actual_width\":" << l.actualGdnWidth()
    << ",\"gdn_packed_width\":" << l.packedGdnWidth
    << ",\"attention_packed_width\":" << l.packedFullWidth
    << ",\"experts\":" << l.experts
    << ",\"experts_per_token\":" << l.expertsPerToken
    << ",\"moe_intermediate_size\":" << l.expertIntermediateSize
    << ",\"shared_expert_intermediate_size\":" << l.expertIntermediateSize
    << ",\"hyper_connection_count\":" << l.hyperConnectionCount
    << ",\"hyper_connection_low_rank\":" << l.hyperConnectionLowRank
    << ",\"indexer_heads\":" << l.indexerHeads
    << ",\"indexer_kv_heads\":" << l.indexerKvHeads
    << ",\"indexer_head_dim\":" << l.indexerHeadDimension
    << ",\"ngram_layer\":" << l.ngramLayer
    << ",\"ngram_vocabulary_size\":" << l.ngramVocabularySize
    << ",\"ngram_embedding_size\":" << l.ngramEmbeddingSize
    << ",\"layer_types\":[";
  for (uint32_t layer = 0; layer < l.layers; ++layer)
    m << (layer ? "," : "")
      << (l.isFullAttentionLayer(layer) ? "\"attention\"" : "\"gdn\"");
  m << "]},\"draft\":{\"architecture\":\"DFlash2DraftModel\",\"layers\":"
    << d.layers << ",\"hidden_size\":" << d.hiddenSize
    << ",\"intermediate_size\":" << d.intermediateSize
    << ",\"sliding_window\":" << ExecutionLimits::draftContextTokens
    << ",\"block_size\":" << ExecutionLimits::draftQueryRows
    << ",\"dynamic_conv_group_size\":16,\"dynamic_conv_kernel_size\":2,"
    << "\"selector_rank\":" << d.selectorRank << ",\"selector_top_k\":16,"
    << "\"target_capture_layers\":[";
  for (size_t index = 0; index < Qwen4ExpLayout::hiddenCaptureLayers.size();
       ++index)
    m << (index ? "," : "") << Qwen4ExpLayout::hiddenCaptureLayers[index];
  m << "]}}";
  std::ofstream(root / "manifest.json") << m.str();

  std::filesystem::create_directories(root / "tokenizer");
  std::ostringstream t;
  t << "{\"text_config\":{\"model_type\":\"qwen4_exp_text\",\"hidden_size\":"
    << l.hiddenSize << ",\"vocab_size\":" << l.vocabularySize
    << ",\"max_position_embeddings\":" << l.maximumContextTokens << "}}";
  std::ofstream(root / "tokenizer" / "config.json") << t.str();
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    require(argc >= 2, "usage: qwen4exp-package <scratch directory>");
    const std::filesystem::path root =
        std::filesystem::path(argv[1]) / "qwen4exp-synthetic";
    std::filesystem::remove_all(root);

    constexpr Qwen4ExpLayout layout;
    DFlashDraftLayout draft;
    draft.layers = 5;
    draft.hiddenSize = layout.hiddenSize;
    draft.dynamicSize = 768;
    draft.qkvSize = 3072;
    draft.attentionSize = 2048;
    draft.intermediateSize = 8704;
    draft.targetHiddenSize = layout.capturedHiddenSize();
    ops::VisionLayout vision;
    vision.outputHiddenSize = layout.hiddenSize;

    uint64_t targetBytes = 0, draftBytes = 0;
    for (uint32_t layer = 0; layer < layout.layers; ++layer) {
      const bool full = layout.isFullAttentionLayer(layer);
      targetBytes += writeWeightFile(
          root / "target" / ("layer-" + std::to_string(layer) + ".bin"),
          Qwen4ExpLayout::layerMagic, layer, full ? 1U : 0U,
          layerSections(layout, full));
    }
    std::vector<uint64_t> head = hyperConnectionSections(layout);
    head.push_back(uint64_t(layout.hiddenSize) * kBFloat16Bytes);
    head.push_back(q4Bytes(layout.vocabularySize, layout.hiddenSize));
    targetBytes += writeWeightFile(root / "target/head.bin", "MDFN0002",
                                   layout.layers, 2, head);
    const uint64_t embeddingElements =
        uint64_t(layout.vocabularySize) * layout.hiddenSize;
    targetBytes += writeWeightFile(
        root / "target/embedding.bin", "MDFN0003", layout.vocabularySize,
        layout.hiddenSize,
        std::array<uint64_t, 3>{embeddingElements / 2, embeddingElements / 32,
                                embeddingElements / 32});
    const uint64_t ngramElements =
        uint64_t(layout.ngramVocabularySize) * layout.ngramEmbeddingSize;
    const uint64_t ngramBytes = writeWeightFile(
        root / "target/ngram.bin", "MDFN0004", layout.ngramVocabularySize,
        layout.ngramEmbeddingSize,
        std::array<uint64_t, 3>{ngramElements / 2, ngramElements / 32,
                                ngramElements / 32});
    targetBytes += ngramBytes;

    for (uint32_t layer = 0; layer < draft.layers; ++layer)
      draftBytes += writeWeightFile(
          root / "draft" / ("layer-" + std::to_string(layer) + ".bin"),
          "MDFD0004", layer, 0, draftLayerSections(draft));
    const uint64_t codebook =
        uint64_t(draft.vocabularySize) * draft.selectorRank * kBFloat16Bytes;
    draftBytes += writeWeightFile(
        root / "draft/model.bin", "MDFD0004", draft.layers, 1,
        std::array<uint64_t, 6>{
            q4Bytes(draft.hiddenSize, draft.targetHiddenSize),
            uint64_t(draft.hiddenSize) * kBFloat16Bytes,
            uint64_t(draft.hiddenSize) * kBFloat16Bytes,
            q4Bytes(draft.selectorRank, draft.hiddenSize), codebook,
            codebook});

    writeManifest(root, layout, draft);

    uint64_t onDisk = 0;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(root))
      if (entry.is_regular_file()) onDisk += entry.file_size();

    std::cout << "synthetic qwen4exp package at " << root << "\n"
              << "  target weights   " << gib(targetBytes) << "\n"
              << "    of which n-gram" << "  " << gib(ngramBytes) << "\n"
              << "  draft weights    " << gib(draftBytes) << "\n"
              << "  apparent total   " << gib(onDisk) << "\n";

    // The manifest branch and the loader, on the real geometry.
    ModelDescriptor descriptor = inspectModelPackage(root);
    require(std::holds_alternative<Qwen4ExpLayout>(descriptor.target),
            "inspectModelPackage did not select the qwen4exp layout");
    require(descriptor.valid(), "qwen4exp descriptor is not self-consistent");
    std::cout << "  manifest         accepted, descriptor valid\n";

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    const uint64_t workingSet = device ? device.recommendedMaxWorkingSetSize : 0;
    std::cout << "  device working set " << gib(workingSet) << "\n";
    if (workingSet > 0 && targetBytes + draftBytes > workingSet) {
      std::cout << "  VERDICT          weights exceed the working set by "
                << gib(targetBytes + draftBytes - workingSet)
                << "\n                   before any cache or activation\n";
    }

    std::filesystem::remove_all(root);
    std::cout << "qwen4exp package test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "qwen4exp package test failed: " << error.what() << '\n';
    return 1;
  }
}
