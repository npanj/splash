#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>
#include <compare>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::ops {

// Immutable views of one packed Q4 projection.  StorageN is part of the
// package ABI; the operator may choose a different compute tile at runtime.
struct Q4Projection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// Q8 affine projections use per-64-input quantization and StorageN=256 order.
// Used by the MoE router and shared-expert gate.
struct Q8Projection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

using VocabularyProjection = std::variant<Q4Projection, Q8Projection>;

// Expert-major Q4 slabs keep one complete StorageN-packed projection per
// expert. The operator selects expertStrideBytes directly; no per-expert
// MetalBuffer objects or weight copies are created at runtime.
struct ExpertQ4Projection final {
  metal::MetalBuffer packed;
  uint32_t experts = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint64_t expertStrideBytes = 0;
};

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
enum class LinearTile : uint8_t { N128, N256, Paired128 };
enum class LinearSimdgroups : uint8_t { Four = 4, Eight = 8 };

struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  auto operator<=>(const LinearWorkload &) const = default;
};

struct LinearConfig final {
  LinearTile tile = LinearTile::N128;
  // Decode grid size. Prefill uses its matrix grid and requires zero here.
  uint32_t groups = 0;
  // Cooperative execution scope, independent of the persistent grid size.
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  bool operator==(const LinearConfig &) const = default;
};

struct LinearChoice final {
  LinearWorkload workload;
  LinearConfig configuration;
};

class LinearPlan final {
public:
  [[nodiscard]] LinearWorkload workload() const noexcept { return workload_; }
  [[nodiscard]] LinearConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t storageRows() const noexcept;
  [[nodiscard]] uint32_t tileColumns() const noexcept;
  [[nodiscard]] uint32_t threadsPerThreadgroup() const noexcept;
  [[nodiscard]] uint64_t sumsBytes() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;
  [[nodiscard]] uint64_t downSumsBytes() const noexcept;
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }

private:
  friend class Q4Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config);
  LinearWorkload workload_;
  LinearConfig config_;
  std::string_view pipeline_;
  std::string_view secondPipeline_;
};

// The plan defines which fields are used and how much scratch they require.
struct LinearBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer sums;
  metal::MetalBuffer residual;
  metal::MetalBuffer gateScratch;
  metal::MetalBuffer downSums;
};

struct Q4DispatchStats final {
  uint64_t fusedSourceOperations = 0;
  uint64_t m16Dispatches = 0;
  uint64_t m24Dispatches = 0;
  uint64_t m32Dispatches = 0;
};

// Owns Q4 pipeline selection and dispatch. Device policy uses GPU family,
// core count and workload tile counts.
class Q4Linear final {
public:
  explicit Q4Linear(const DeviceCapabilities &device) noexcept;

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config);
  [[nodiscard]] std::vector<LinearPlan> candidates(LinearWorkload workload) const;
  // Installed only at startup; encoding does a read-only lookup, never tuning.
  void setChoices(std::span<const LinearChoice> choices);
  void add(metal::CommandGraph &graph, LinearBuffers buffers,
           const Q4Projection &projection, const LinearPlan &plan,
           const Q4Projection *gate = nullptr,
           Q4DispatchStats *stats = nullptr) const;

  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer sums, LinearMatrix matrix,
                      uint32_t rows) const;
  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const Q4Projection &projection, metal::MetalBuffer output,
                  metal::MetalBuffer sums, LinearMatrix matrix,
                  uint32_t rows) const;
  void addPrefillUpWithGate(
      metal::CommandGraph &graph, metal::MetalBuffer input,
      const Q4Projection &up, metal::MetalBuffer gateScratch,
      metal::MetalBuffer output, metal::MetalBuffer sums,
      metal::MetalBuffer downSums, LinearMatrix matrix,
      uint32_t rows) const;
  void addPrefillResidual(metal::CommandGraph &graph,
                          metal::MetalBuffer input,
                          const Q4Projection &projection,
                          metal::MetalBuffer residual,
                          metal::MetalBuffer output, metal::MetalBuffer sums,
                          LinearMatrix matrix, uint32_t rows) const;

  void addDecode(metal::CommandGraph &graph,
                 metal::MetalBuffer input, const Q4Projection &projection,
                 metal::MetalBuffer output, LinearMatrix matrix) const;
  void addDecodeBatch(metal::CommandGraph &graph,
                      metal::MetalBuffer input,
                      const Q4Projection &projection,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats) const;
  void addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const Q4Projection &gate, const Q4Projection &up,
                      metal::MetalBuffer gateScratch,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats) const;
  void addResidualBatch(metal::CommandGraph &graph,
                        metal::MetalBuffer input,
                        const Q4Projection &projection,
                        metal::MetalBuffer residual,
                        metal::MetalBuffer output, LinearMatrix matrix,
                        uint32_t lanes, Q4DispatchStats &stats) const;

  void add(metal::CommandGraph &graph, LinearBuffers buffers,
           const Q8Projection &projection, const LinearPlan &plan,
           const Q8Projection *gate = nullptr,
           Q4DispatchStats *stats = nullptr) const;
  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const Q8Projection &projection, metal::MetalBuffer output,
                  metal::MetalBuffer sums, LinearMatrix matrix,
                  uint32_t rows) const;
  void addPrefillUpWithGate(
      metal::CommandGraph &graph, metal::MetalBuffer input,
      const Q8Projection &up, metal::MetalBuffer gateScratch,
      metal::MetalBuffer output, metal::MetalBuffer sums,
      metal::MetalBuffer downSums, LinearMatrix matrix,
      uint32_t rows) const;
  void addPrefillResidual(metal::CommandGraph &graph,
                          metal::MetalBuffer input,
                          const Q8Projection &projection,
                          metal::MetalBuffer residual,
                          metal::MetalBuffer output, metal::MetalBuffer sums,
                          LinearMatrix matrix, uint32_t rows) const;
  void addDecode(metal::CommandGraph &graph,
                 metal::MetalBuffer input, const Q8Projection &projection,
                 metal::MetalBuffer output, LinearMatrix matrix) const;
  void addDecodeBatch(metal::CommandGraph &graph,
                      metal::MetalBuffer input,
                      const Q8Projection &projection,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats) const;
  void addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const Q8Projection &gate, const Q8Projection &up,
                      metal::MetalBuffer gateScratch,
                      metal::MetalBuffer output, LinearMatrix matrix,
                      uint32_t lanes, Q4DispatchStats &stats) const;
  void addResidualBatch(metal::CommandGraph &graph,
                        metal::MetalBuffer input,
                        const Q8Projection &projection,
                        metal::MetalBuffer residual,
                        metal::MetalBuffer output, LinearMatrix matrix,
                        uint32_t lanes, Q4DispatchStats &stats) const;

private:
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload) const;
  uint32_t appleGpuFamily_ = 0;
  uint32_t gpuCores_ = 0;
  std::vector<LinearChoice> choices_;
};

} // namespace splash::ops
