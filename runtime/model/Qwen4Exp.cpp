#include "Qwen4Exp.hpp"

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kNextHeadMagic = "MDFN0002";
constexpr std::string_view kNextEmbeddingMagic = "MDFN0003";
constexpr std::string_view kNextNgramMagic = "MDFN0004";

void requireLayout(const Qwen4ExpLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize || !layout.expertStorageN ||
      !layout.hyperConnectionCount || !layout.hyperConnectionLowRank ||
      !layout.indexerHeads || !layout.indexerKvHeads ||
      !layout.indexerHeadDimension || !layout.ngramVocabularySize ||
      !layout.ngramEmbeddingSize) {
    throw WeightStoreError("qwen4exp layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.gdnValueHeads * layout.gdnHeadDimension ||
      layout.packedFullWidth != layout.actualFullWidth() ||
      layout.packedGdnWidth < layout.actualGdnWidth() ||
      layout.packedGdnWidth % kQ4StorageN ||
      layout.expertsPerToken > layout.experts ||
      layout.ngramLayer >= layout.layers ||
      layout.hiddenCaptureLayers.back() >= layout.layers ||
      !layout.q8Layout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("qwen4exp layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  // Experts and the indexer tile narrower; both are multiples of 128 only.
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.indexerProjectionWidth(), layout.hiddenSize,
                   layout.expertStorageN);
}

Qwen4ExpHyperConnection readHyperConnection(WeightFile &file,
                                            const Qwen4ExpLayout &layout,
                                            std::string_view label) {
  const uint64_t width =
      checkedWeightMultiply(layout.hyperConnectionWidth(), kBFloat16Bytes,
                            "hyper-connection width bytes");
  const std::string prefix(label);
  return {
      file.section(width, prefix + "-norm"),
      file.section(checkedWeightMultiply(layout.hyperConnectionCount, width,
                                         "hyper-connection inject bytes"),
                   prefix + "-inject"),
      file.section(checkedWeightMultiply(layout.hyperConnectionLowRank, width,
                                         "hyper-connection mix bytes"),
                   prefix + "-mix-down"),
      file.section(checkedWeightMultiply(layout.hyperConnectionLowRank, width,
                                         "hyper-connection mix bytes"),
                   prefix + "-mix-up"),
  };
}

Qwen4ExpIndexer readIndexer(WeightFile &file, metal::MetalBackend &backend,
                            const Qwen4ExpLayout &layout) {
  const uint64_t normBytes =
      checkedWeightMultiply(layout.indexerHeadDimension, kBFloat16Bytes,
                            "indexer norm bytes");
  ops::Q4Projection projection = readQ4Projection(
      file, backend, layout.indexerProjectionWidth(), layout.hiddenSize,
      "indexer-qk");
  return {projection, file.section(normBytes, "indexer-query-norm"),
          file.section(normBytes, "indexer-key-norm")};
}

void readExperts(WeightFile &file, metal::MetalBackend &backend,
                 const Qwen4ExpLayout &layout, ops::MoeWeights &ffn) {
  const uint32_t n = layout.expertStorageN;
  ffn.router = readQ8Projection(file, backend, layout.experts,
                                layout.hiddenSize, "router");
  ffn.expertGate = readExpertQ4Projection(
      file, layout.experts, layout.expertIntermediateSize, layout.hiddenSize,
      "experts-gate", n);
  ffn.expertUp = readExpertQ4Projection(
      file, layout.experts, layout.expertIntermediateSize, layout.hiddenSize,
      "experts-up", n);
  ffn.expertDown = readExpertQ4Projection(
      file, layout.experts, layout.hiddenSize, layout.expertIntermediateSize,
      "experts-down", n);
  ffn.sharedGate = readExpertQ4Projection(
      file, 1, layout.expertIntermediateSize, layout.hiddenSize,
      "shared-expert-gate", n);
  ffn.sharedUp = readExpertQ4Projection(
      file, 1, layout.expertIntermediateSize, layout.hiddenSize,
      "shared-expert-up", n);
  ffn.sharedDown = readExpertQ4Projection(
      file, 1, layout.hiddenSize, layout.expertIntermediateSize,
      "shared-expert-down", n);
  ffn.sharedExpertGate = readQ8Projection(
      file, backend, kQ4StorageN, layout.hiddenSize,
      "shared-expert-scalar-gate");
}

} // namespace

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend,
                                    const std::filesystem::path &directory,
                                    Qwen4ExpLayout layout) {
  requireLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Qwen4ExpWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    const std::string filename =
        "layer-" + std::to_string(layerIndex) + ".bin";
    WeightFile file(backend, directory / filename, "target/" + filename,
                    Qwen4ExpLayout::layerMagic, layerIndex,
                    fullAttention ? 1U : 0U);
    auto &layer = result.layers.emplace_back();
    layer.attentionHyperConnection =
        readHyperConnection(file, layout, "attention-hyper");
    layer.mixer =
        readQwenMixer(file, backend, layout.mixerGeometry(), fullAttention);
    if (fullAttention) layer.indexer = readIndexer(file, backend, layout);
    layer.mlpHyperConnection = readHyperConnection(file, layout, "mlp-hyper");
    readExperts(file, backend, layout, layer.ffn);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file(backend, directory / "head.bin", "target/head.bin",
                    kNextHeadMagic, layout.layers, 2);
    result.hyperConnectionMixer =
        readHyperConnection(file, layout, "hyper-mixer");
    result.finalNorm = file.section(
        checkedWeightMultiply(layout.hiddenSize, kBFloat16Bytes,
                              "qwen4exp norm bytes"),
        "final-norm");
    result.logitsProjection = readQ4Projection(
        file, backend, layout.vocabularySize, layout.hiddenSize, "logits");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file(backend, directory / "embedding.bin",
                    "target/embedding.bin", kNextEmbeddingMagic,
                    layout.vocabularySize, layout.hiddenSize);
    result.tokenEmbedding = readQ4ProjectionComponents(
        file, layout.vocabularySize, layout.hiddenSize, "embedding");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    // Gathered per token like the embedding, so stored the same way and
    // kept out of the layer files it belongs to.
    WeightFile file(backend, directory / "ngram.bin", "target/ngram.bin",
                    kNextNgramMagic, layout.ngramVocabularySize,
                    layout.ngramEmbeddingSize);
    result.ngramEmbedding = readQ4ProjectionComponents(
        file, layout.ngramVocabularySize, layout.ngramEmbeddingSize, "ngram");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
