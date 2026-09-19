#include "model/Qwen4Exp.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightStore.hpp"

#include <iostream>
#include <stdexcept>

namespace {
using namespace splash;
using namespace splash::model;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Mirrors validateQ4Layout, which lives in WeightStore.cpp and would pull
// Metal into what is otherwise a pure arithmetic test.
constexpr bool q4LayoutIsValid(uint32_t outputSize, uint32_t inputSize,
                               uint32_t storageN = kQ4StorageN) {
  return outputSize && inputSize && inputSize % kQ4GroupElements == 0 &&
         outputSize % storageN == 0;
}

// The packed widths are derived, not transcribed: a concatenation padded up
// to StorageN. Checking them here catches a bad constant before any weight
// file is written against it.
void packedWidthsFollowFromTheConcatenation() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.actualGdnWidth() == 16480,
                "qkv + z + b + a must be 16480 rows");
  static_assert(layout.packedGdnWidth == 16640,
                "16480 padded up to StorageN 256 is 16640");
  static_assert(layout.packedGdnWidth % kQ4StorageN == 0);
  static_assert(layout.packedGdnWidth - layout.actualGdnWidth() == 160,
                "the last GDN tile carries 160 zero rows");

  static_assert(layout.actualFullWidth() == 13312,
                "2*q + k + v with a gated query is 13312 rows");
  static_assert(layout.packedFullWidth == layout.actualFullWidth(),
                "the attention concatenation needs no padding");
  static_assert(layout.packedFullWidth % kQ4StorageN == 0);

  static_assert(layout.attentionWidth ==
                    layout.gdnValueHeads * layout.gdnHeadDimension,
                "attention width is value heads by head dimension");
  static_assert(layout.convolutionDimension ==
                    2 * layout.gdnKeyHeads * layout.gdnHeadDimension +
                        layout.gdnValueHeads * layout.gdnHeadDimension,
                "conv width is 2*k + v head widths");
}

// Three layers in four are linear attention, and their mixer geometry is
// identical to the Qwen3.8 GDN block apart from the hidden size. This is the
// reason the existing GDN kernels carry most of the model.
void linearAttentionMatchesQwen38() {
  constexpr Qwen4ExpLayout next;
  constexpr Qwen3_8Layout dense;

  static_assert(next.convolutionDimension == dense.convolutionDimension);
  static_assert(next.gdnKeyHeads == dense.gdnKeyHeads);
  static_assert(next.gdnValueHeads == dense.gdnValueHeads);
  static_assert(next.gdnHeadDimension == dense.gdnHeadDimension);
  static_assert(next.attentionWidth == dense.attentionWidth);
  static_assert(next.attentionHeadDimension == dense.attentionHeadDimension);
  static_assert(next.attentionQueryHeads == dense.attentionQueryHeads);
  static_assert(next.fullAttentionPeriod == dense.fullAttentionPeriod);
  static_assert(next.vocabularySize == dense.vocabularySize);
  static_assert(next.packedGdnWidth == dense.packedGdnWidth,
                "both models pad the GDN concatenation to the same width");

  // 48 layers at period 4 gives 12 attention layers and 36 linear ones.
  static_assert(next.attentionLayerCount() == 12);
  static_assert(next.layers - next.attentionLayerCount() == 36);
  static_assert(!next.isFullAttentionLayer(0));
  static_assert(next.isFullAttentionLayer(3));
  static_assert(next.isFullAttentionLayer(47));
}

// Every projection the engine can already express must satisfy the Q4 rules.
void supportedProjectionsAreQ4Aligned() {
  constexpr Qwen4ExpLayout layout;
  static_assert(q4LayoutIsValid(layout.packedGdnWidth, layout.hiddenSize));
  static_assert(q4LayoutIsValid(layout.packedFullWidth, layout.hiddenSize));
  static_assert(q4LayoutIsValid(layout.hiddenSize, layout.attentionWidth));
  static_assert(q4LayoutIsValid(layout.vocabularySize, layout.hiddenSize));
  require(layout.hiddenSize % kQ4GroupElements == 0,
          "hidden size must be group-aligned as a projection input");
  require(layout.hiddenSize % kQ4StorageN == 0,
          "hidden size must be StorageN-aligned as a projection output");
}

// Expert projections tile narrower than everything else. This is the whole
// reason kQ4ExpertStorageN exists, so pin both directions.
void expertProjectionsTileAt128() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.expertStorageN == kQ4ExpertStorageN);

  // 640 is not expressible at the usual width, which is why it tiles narrower.
  static_assert(!q4LayoutIsValid(layout.expertIntermediateSize,
                                 layout.hiddenSize, kQ4StorageN),
                "640 must not be expressible at StorageN 256");
  static_assert(q4LayoutIsValid(layout.expertIntermediateSize,
                                layout.hiddenSize, kQ4ExpertStorageN),
                "640 must be expressible at StorageN 128");
  static_assert(q4LayoutIsValid(layout.hiddenSize,
                                layout.expertIntermediateSize,
                                kQ4ExpertStorageN),
                "the down projection must tile at 128 as well");

  // The narrower tile must still be a whole number of 32-wide matmul slices.
  static_assert(kQ4ExpertStorageN % 32 == 0);
  static_assert(kQ4StorageN % kQ4ExpertStorageN == 0);

  // What padding to the usual width would have cost, for the record: three
  // projections per expert, 512 experts, 48 layers, at 9 bytes per 16 weights.
  constexpr uint64_t stored =
      uint64_t(layout.expertIntermediateSize) * layout.hiddenSize * 3 *
      layout.experts * layout.layers * 9 / 16;
  constexpr uint64_t padded = uint64_t(768) * layout.hiddenSize * 3 *
                              layout.experts * layout.layers * 9 / 16;
  require(padded - stored > 12ULL * 1024 * 1024 * 1024,
          "padding to 768 would have cost more than 12 GiB");
}

// The one gap that remains, asserted rather than described, so that lifting
// the limit fails this test and forces the constant here to be revisited.
void expertCountIsStillUnsupported() {
  constexpr Qwen4ExpLayout layout;
  require(layout.experts == kUnsupportedExpertCount, "expert count changed");
  require(layout.experts > 256,
          "ops::MoE accepts at most 256 experts; update this test if lifted");
}

void stateLayoutsAreConsistent() {
  constexpr Qwen4ExpLayout layout;
  const auto kv = layout.q8Layout();
  require(kv.attentionLayers == 12,
          "KV cache covers only the attention layers");
  require(kv.kvHeads == layout.attentionKvHeads, "KV heads mismatch");
  require(kv.headDimension == layout.attentionHeadDimension,
          "KV head dimension mismatch");

  const auto gdn = layout.gdnStateLayout();
  require(gdn.layers == 36, "GDN state covers only the linear layers");
  require(gdn.convolutionChannels == layout.convolutionDimension,
          "GDN conv channel count mismatch");
  require(gdn.convolutionHistory == kGdnConvolutionTaps - 1,
          "GDN conv history must be taps minus one");

  require(layout.capturedHiddenSize() ==
              layout.hiddenSize * Qwen4ExpLayout::hiddenCaptureLayers.size(),
          "captured hidden size must follow the capture layer count");
  for (uint32_t index : Qwen4ExpLayout::hiddenCaptureLayers) {
    require(index < layout.layers, "a capture layer is out of range");
  }
}

} // namespace

int main() {
  try {
    packedWidthsFollowFromTheConcatenation();
    linearAttentionMatchesQwen38();
    supportedProjectionsAreQ4Aligned();
    expertProjectionsTileAt128();
    expertCountIsStillUnsupported();
    stateLayoutsAreConsistent();
  } catch (const std::exception &error) {
    std::cerr << "qwen4exp layout test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "qwen4exp layout test passed\n";
  return 0;
}
