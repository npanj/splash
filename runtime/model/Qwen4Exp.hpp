#pragma once

#include "QwenTarget.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace splash::model {

// Qwen3.8-Flash-Next, upstream architecture name `qwen4exp`. Its linear
// attention is dimensionally identical to the Qwen3.8 GDN block, so three
// layers in four reuse the existing mixer geometry; the fourth is sparse
// attention with a routing indexer, which has no operator yet.
//
// Numbers below come from the published config.json. The fields under
// "no Splash equivalent" describe modules this engine cannot execute today;
// they are recorded here so the package validator can check them and so the
// gaps are visible in one place. See kUnsupported* below.
struct Qwen4ExpLayout final {
  static constexpr std::string_view layerMagic = "MDFN0001";

  // Provisional: the draft is deferred, so these five indices are evenly
  // spaced rather than chosen against a trained DFlash 2 draft.
  static constexpr std::array<uint32_t, 5> hiddenCaptureLayers{3, 13, 23, 33, 43};

  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 48;
  uint32_t hiddenSize = 2560;
  uint32_t vocabularySize = 248320;
  uint32_t packedGdnWidth = 16640;
  uint32_t packedFullWidth = 13312;
  uint32_t convolutionDimension = 10240;
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 48;
  uint32_t gdnHeadDimension = 128;
  uint32_t attentionWidth = 6144;
  uint32_t attentionQueryHeads = 24;
  uint32_t attentionKvHeads = 2;
  uint32_t attentionHeadDimension = 256;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  uint32_t fullAttentionPeriod = 4;
  uint32_t experts = 512;
  uint32_t expertsPerToken = 10;
  uint32_t expertIntermediateSize = 640;
  // 640 is not a multiple of 256 but is a multiple of 128, so the expert
  // projections tile narrower than everything else rather than padding to
  // 768. See kQ4ExpertStorageN.
  uint32_t expertStorageN = kQ4ExpertStorageN;

  // TODO(qwen4exp): confirm against the shipped tokenizer; the vocabulary
  // matches Qwen3.8 exactly, so these are carried over as placeholders.
  uint32_t maskToken = 248070;
  std::array<uint32_t, 2> stopTokens{248044, 248046};

  // ---- no Splash equivalent; recorded, not executable -------------------
  uint32_t hyperConnectionCount = 4;
  uint32_t hyperConnectionLowRank = 320;
  uint32_t indexerHeads = 4;
  uint32_t indexerHeadDimension = 128;
  uint32_t indexerBudget = 2048;
  uint32_t ngramSize = 3;
  uint32_t ngramLayer = 1;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return fullAttentionPeriod && (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return fullAttentionPeriod ? layers / fullAttentionPeriod : 0;
  }
  // Packed projections are concatenated then padded up to StorageN.
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept {
    return convolutionDimension + attentionWidth + 2 * gdnValueHeads;
  }
  // Query width is doubled for the output gate, as in Qwen3.8.
  [[nodiscard]] constexpr uint32_t actualFullWidth() const noexcept {
    return 2 * attentionQueryHeads * attentionHeadDimension +
           2 * attentionKvHeads * attentionHeadDimension;
  }
  [[nodiscard]] constexpr kv::Q8Layout q8Layout() const noexcept {
    return {attentionLayerCount(), attentionKvHeads, attentionHeadDimension};
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), kGdnConvolutionTaps - 1,
            convolutionDimension, gdnValueHeads, gdnHeadDimension,
            gdnHeadDimension};
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }

  bool operator==(const Qwen4ExpLayout &) const = default;
};

// One constraint this architecture still breaks. It is an engine limit, not a
// property of the weights, and it must be lifted before a package loads:
// ops::MoE accepts at most 256 experts, and the router kernel stages 256
// scores per row before sorting them.
//
// The expert width no longer belongs on this list. 640 is expressible exactly
// at kQ4ExpertStorageN = 128; what remains is a MoE kernel instantiated at
// that width, which is Stage 3 work.
inline constexpr uint32_t kUnsupportedExpertCount = 512;

} // namespace splash::model
