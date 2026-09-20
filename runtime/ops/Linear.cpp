#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kPrefillRows = 32;
constexpr uint32_t kQuantGroup = 64;
static_assert(sizeof(LinearMatrix) == 8);

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid Q4 linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid Q4 prefill workload");
  } else if (w.phase == LinearPhase::Decode) {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid Q4 decode workload");
  } else {
    throw std::invalid_argument("invalid Q4 linear phase");
  }
  if (w.epilogue != LinearEpilogue::None && w.epilogue != LinearEpilogue::Residual &&
      w.epilogue != LinearEpilogue::GateUp && w.epilogue != LinearEpilogue::UpWithGate)
    throw std::invalid_argument("invalid Q4 linear epilogue");
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument("Q4 buffer is below plan requirement");
}

void requireProjection(const Q4Projection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("Q4 projection does not match plan");
  requireBytes(p.weights, uint64_t{matrix.outputSize} * matrix.inputSize / 2);
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.scales, bytes);
  requireBytes(p.biases, bytes);
}

void requireProjection(const Q8Projection &p, LinearMatrix matrix) {
  if (p.outputSize != matrix.outputSize || p.inputSize != matrix.inputSize)
    throw std::invalid_argument("Q8 projection does not match plan");
  requireBytes(p.weights, uint64_t{matrix.outputSize} * matrix.inputSize);
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.scales, bytes);
  requireBytes(p.biases, bytes);
}

std::string q8PipelineName(std::string_view q4Name) {
  std::string name(q4Name);
  const std::string from = "_q4_";
  const std::string to = "_q8_";
  size_t pos = name.find(from);
  if (pos != std::string::npos) {
    name.replace(pos, from.length(), to);
  }
  return name;
}

void account(Q4DispatchStats &stats, uint32_t lanes, uint32_t count) noexcept {
  if (lanes == 1) return;
  stats.fusedSourceOperations += uint64_t{lanes} * count;
  if (lanes == 2) stats.m16Dispatches += count;
  else if (lanes == 3) stats.m24Dispatches += count;
  else stats.m32Dispatches += count;
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid Q4 decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// The four-simdgroup kernels: every prefill N128 tile, and the decode M24
// N128 plain and residual projections.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

} // namespace

uint32_t LinearPlan::storageRows() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? ((workload_.rows + kPrefillRows - 1) / kPrefillRows) * kPrefillRows : workload_.rows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  return config_.tile == LinearTile::N256 ? 256 : 128;
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  return static_cast<uint32_t>(config_.simdgroups) * 32;
}
uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp && !secondPipeline_.empty());
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config)
    : workload_(w), config_(config) {
  validate(w);
  if (config.tile != LinearTile::N128 && config.tile != LinearTile::N256 &&
      config.tile != LinearTile::Paired128)
    throw std::invalid_argument("invalid Q4 linear tile");
  if ((config.simdgroups != LinearSimdgroups::Four &&
       config.simdgroups != LinearSimdgroups::Eight) ||
      (config.simdgroups == LinearSimdgroups::Four &&
       !supportsFourSimdgroups(w, config.tile)))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (w.matrix.outputSize % tileColumns())
    throw std::invalid_argument("Q4 matrix is not divisible by tile columns");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (config.groups || config.tile == LinearTile::Paired128)
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_n128_up_silu_sums_sg4"
          : residual ? "prefill_linear_q4_n128_residual_sg4" : "prefill_linear_q4_n128_sg4";
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = "prefill_linear_q4_n256_up_silu_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128_residual" : "prefill_linear_q4_n256_residual";
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128" : "prefill_linear_q4_n256";
    }
    return;
  }
  if (!config.groups || config.groups > w.matrix.outputSize / tileColumns())
    throw std::invalid_argument("invalid Q4 decode group count");
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  if (config.tile == LinearTile::Paired128 && (lane != 0 || w.matrix.outputSize % 256))
    throw std::invalid_argument("paired Q4 tile requires one lane and paired columns");
  if (four) {
    pipeline_ = residual ? "decode_linear_q4_n128_residual_m24_sg4"
                         : "decode_linear_q4_n128_m24_sg4";
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    constexpr std::array names{"decode_linear_q4_n256_gate_up", "decode_linear_q4_n256_gate_up_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
    if (lane >= 2)
      secondPipeline_ = lane == 2 ? "decode_linear_q4_n256_up_silu_m24"
                                  : "decode_linear_q4_n256_up_silu_m32";
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{"decode_linear_q4_n128_residual", "decode_linear_q4_n128_residual_m16",
        "decode_linear_q4_n128_residual_m24", "decode_linear_q4_n128_residual_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
        ? "decode_linear_q4_n128_residual_paired" : names[lane];
  } else if (config.tile == LinearTile::N256) {
    constexpr std::array names{"decode_linear_q4_n256", "decode_linear_q4_n256_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
  } else {
    constexpr std::array names{"decode_linear_q4_n128", "decode_linear_q4_n128_m16",
        "decode_linear_q4_n128_m24", "decode_linear_q4_n128_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? "decode_linear_q4_n128_paired" : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one resident wave for
// longer chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Resident groups per core: one wave for this kernel's register footprint.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// Resident-wave and full-grid thresholds measured on 16/20-core Apple10 GPUs.
// Gate/up uses the conservative limit shared by both devices. Its many-wave
// threshold follows N256; the four-simdgroup threshold scales from N128. Those
// two extrapolations remain unmeasured.
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};
// Apple9 retains its measured gate/up clamp. The round-robin policy above was
// measured on Apple10; applying it to Apple9 requires separate calibration.
constexpr double kApple9GateUpGroupsPerCore = 2.25;

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}
// A multi-row N256 decode tile halves the input re-reads of N128 but also
// halves the grid; it pays only while the N256 grid keeps two tiles per core.
constexpr uint32_t kWideDecodeTilesPerCore = 2;
// Apple9 N256 prefill needs eight threadgroups per core to amortize its larger
// tile. Apple10 selects four-simdgroup N128; that variant is unmeasured on Apple9.
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;
// Without a core count the policy assumes a large GPU, so every rule picks
// the configuration with more threadgroups, which is the safe direction.
constexpr uint32_t kAssumedGpuCores = 64;

} // namespace

Q4Linear::Q4Linear(const DeviceCapabilities &device) noexcept
    : appleGpuFamily_(device.appleGpuFamily),
      gpuCores_(device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores) {}

// GPU family selects variants; core count and workload tile counts determine
// parallelism.
LinearConfig Q4Linear::baseline(LinearWorkload w) const {
  validate(w);
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    if (appleGpuFamily_ >= 10)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kPrefillRows - 1) / kPrefillRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * gpuCores_;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  // Apple9 keeps its one-tile grids (see kApple9GateUpGroupsPerCore).
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return appleGpuFamily_ >= 10 ? decodeGroups(tiles, gpuCores_, policy)
                                 : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (appleGpuFamily_ < 10) {
      const auto resident = static_cast<uint32_t>(
          std::max(1L, std::lround(kApple9GateUpGroupsPerCore * gpuCores_)));
      return {LinearTile::N256, std::min(tiles256, resident)};
    }
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  }
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, kN128Groups)};
  // M24 plain projections benefit from four SIMD groups on Apple9 too.
  // Apple9 residual projections retain eight groups with compact prefix
  // traversal; Apple10 uses four groups for both epilogues.
  if (lanes == 3 && (appleGpuFamily_ >= 10 ||
      (appleGpuFamily_ == 9 && w.epilogue == LinearEpilogue::None)))
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_)
    return {LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

LinearPlan Q4Linear::plan(LinearWorkload workload) const {
  const auto found = std::lower_bound(choices_.begin(), choices_.end(), workload,
      [](const LinearChoice &choice, LinearWorkload key) { return choice.workload < key; });
  return LinearPlan(workload, found != choices_.end() && found->workload == workload
      ? found->configuration : baseline(workload));
}
LinearPlan Q4Linear::plan(LinearWorkload workload, LinearConfig config) {
  return LinearPlan(workload, config);
}
void Q4Linear::setChoices(std::span<const LinearChoice> choices) {
  std::vector<LinearChoice> pending(choices.begin(), choices.end());
  for (const auto &choice : pending) (void)plan(choice.workload, choice.configuration);
  std::sort(pending.begin(), pending.end(), [](const auto &a, const auto &b) {
    return a.workload < b.workload;
  });
  for (size_t i = 1; i < pending.size(); ++i)
    if (pending[i - 1].workload == pending[i].workload)
      throw std::invalid_argument("duplicate Q4 linear choice");
  choices_ = std::move(pending);
}

std::vector<LinearPlan> Q4Linear::candidates(LinearWorkload w) const {
  std::vector<LinearPlan> result;
  result.reserve(16);
  result.push_back(LinearPlan(w, baseline(w)));
  const auto append = [&](LinearConfig config) {
    for (const auto &existing : result)
      if (existing.configuration() == config) return;
    result.push_back(LinearPlan(w, config));
  };
  for (const auto tile : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128}) {
    const uint32_t columns = tile == LinearTile::N256 ? 256 : 128;
    if (w.matrix.outputSize % columns ||
        (tile == LinearTile::Paired128 && (w.phase != LinearPhase::Decode ||
         w.rows != SPLASH_TARGET_VERIFY_ROWS || w.matrix.outputSize % 256)) ||
        (w.epilogue == LinearEpilogue::GateUp && tile != LinearTile::N256) ||
        (w.phase == LinearPhase::Decode && w.epilogue == LinearEpilogue::Residual && tile == LinearTile::N256))
      continue;
    if (w.phase == LinearPhase::Prefill) {
      // The fused up projection has no eight-simdgroup N128 kernel.
      if (tile == LinearTile::N256 || w.epilogue != LinearEpilogue::UpWithGate)
        append({tile, 0});
      if (supportsFourSimdgroups(w, tile))
        append({tile, 0, LinearSimdgroups::Four});
    } else {
      const uint32_t tiles = w.matrix.outputSize / columns;
      for (const uint32_t groups : {36U, 60U, 80U, tiles}) {
        append({tile, std::min(groups, tiles)});
        if (supportsFourSimdgroups(w, tile))
          append({tile, std::min(groups, tiles), LinearSimdgroups::Four});
      }
    }
  }
  return result;
}

void Q4Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Q4Projection &p, const LinearPlan &selected, const Q4Projection *gate,
    Q4DispatchStats *stats) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  requireProjection(p, w.matrix);
  requireBytes(b.input, uint64_t{selected.storageRows()} * k * 2);
  requireBytes(b.output, uint64_t{selected.storageRows()} * n * 2);
  requireBytes(b.sums, selected.sumsBytes());
  requireBytes(b.gateScratch, selected.gateScratchBytes());
  requireBytes(b.downSums, selected.downSumsBytes());
  if (w.epilogue == LinearEpilogue::Residual)
    requireBytes(b.residual, uint64_t{selected.storageRows()} * n * 2);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate) throw std::invalid_argument("Q4 gate projection is missing");
    requireProjection(*gate, w.matrix);
  } else if (gate) throw std::invalid_argument("unexpected Q4 gate projection");
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill)
      graph.add(std::string(name), bindings,
          Q4PrefillParams{w.matrix.outputSize, w.matrix.inputSize},
          {selected.storageRows() / kPrefillRows, n / selected.tileColumns(), 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    else {
      const uint32_t groups = selected.configuration().groups;
      graph.add(std::string(name), bindings, Q4Params{n, k, groups}, {groups, 1, 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    }
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases,
          b.output, p.weights, p.scales, p.biases});
    else {
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases, b.gateScratch});
      dispatch(selected.secondPipeline(), {b.input, p.weights, p.scales, p.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
        b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (w.phase == LinearPhase::Prefill)
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output, b.sums});
    else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output});
  } else if (w.phase == LinearPhase::Prefill)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output, b.sums});
  else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output});
  if (stats && w.phase == LinearPhase::Decode)
    account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, selected.secondPipeline().empty() ? 1 : 2);
}

void Q4Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  validate({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None});
  const uint32_t tiles = (rows + kPrefillRows - 1) / kPrefillRows;
  const uint64_t storageRows = uint64_t{tiles} * kPrefillRows;
  requireBytes(input, storageRows * matrix.inputSize * 2);
  requireBytes(sums, storageRows * (matrix.inputSize / kQuantGroup) * 4);
  graph.add("prefill_linear_q4_sums32", {input, sums},
      Q4PrefillParams{matrix.outputSize, matrix.inputSize}, {tiles, 1, 1});
}
void Q4Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, metal::MetalBuffer sums,
    LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None}));
}
void Q4Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, residual, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::Residual}));
}
void Q4Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &up, metal::MetalBuffer gateScratch, metal::MetalBuffer output,
    metal::MetalBuffer sums, metal::MetalBuffer downSums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, gateScratch, downSums}, up,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate}));
}
void Q4Linear::addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, LinearMatrix matrix) const {
  add(graph, {input, output, {}, {}, {}, {}}, p, plan(decode(matrix, 1, LinearEpilogue::None)));
}
void Q4Linear::addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer output, LinearMatrix matrix,
    uint32_t lanes, Q4DispatchStats &stats) const {
  add(graph, {input, output, {}, {}, {}, {}}, p, plan(decode(matrix, lanes, LinearEpilogue::None)), nullptr, &stats);
}
void Q4Linear::addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats) const {
  add(graph, {input, output, {}, residual, {}, {}}, p, plan(decode(matrix, lanes, LinearEpilogue::Residual)), nullptr, &stats);
}
void Q4Linear::addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q4Projection &gate, const Q4Projection &up, metal::MetalBuffer gateScratch,
    metal::MetalBuffer output, LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats) const {
  add(graph, {input, output, {}, {}, gateScratch, {}}, up, plan(decode(matrix, lanes, LinearEpilogue::GateUp)), &gate, &stats);
}

void Q4Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Q8Projection &p, const LinearPlan &selected, const Q8Projection *gate,
    Q4DispatchStats *stats) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  requireProjection(p, w.matrix);
  requireBytes(b.input, uint64_t{selected.storageRows()} * k * 2);
  requireBytes(b.output, uint64_t{selected.storageRows()} * n * 2);
  requireBytes(b.sums, selected.sumsBytes());
  requireBytes(b.gateScratch, selected.gateScratchBytes());
  requireBytes(b.downSums, selected.downSumsBytes());
  if (w.epilogue == LinearEpilogue::Residual)
    requireBytes(b.residual, uint64_t{selected.storageRows()} * n * 2);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate) throw std::invalid_argument("Q8 gate projection is missing");
    requireProjection(*gate, w.matrix);
  } else if (gate) throw std::invalid_argument("unexpected Q8 gate projection");
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    const std::string q8Name = q8PipelineName(name);
    if (w.phase == LinearPhase::Prefill)
      graph.add(q8Name, bindings,
          Q4PrefillParams{w.matrix.outputSize, w.matrix.inputSize},
          {selected.storageRows() / kPrefillRows, n / selected.tileColumns(), 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    else {
      const uint32_t groups = selected.configuration().groups;
      graph.add(q8Name, bindings, Q4Params{n, k, groups}, {groups, 1, 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    }
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases,
          b.output, p.weights, p.scales, p.biases});
    else {
      dispatch(selected.pipeline(), {b.input, gate->weights, gate->scales, gate->biases, b.gateScratch});
      dispatch(selected.secondPipeline(), {b.input, p.weights, p.scales, p.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases,
        b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (w.phase == LinearPhase::Prefill)
      dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output, b.sums});
    else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.residual, b.output});
  } else if (w.phase == LinearPhase::Prefill)
    dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output, b.sums});
  else dispatch(selected.pipeline(), {b.input, p.weights, p.scales, p.biases, b.output});
  if (stats && w.phase == LinearPhase::Decode)
    account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, selected.secondPipeline().empty() ? 1 : 2);
}

void Q4Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &p, metal::MetalBuffer output, metal::MetalBuffer sums,
    LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::None}));
}

void Q4Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    metal::MetalBuffer sums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, residual, {}, {}}, p,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::Residual}));
}

void Q4Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &up, metal::MetalBuffer gateScratch, metal::MetalBuffer output,
    metal::MetalBuffer sums, metal::MetalBuffer downSums, LinearMatrix matrix, uint32_t rows) const {
  add(graph, {input, output, sums, {}, gateScratch, downSums}, up,
      plan({matrix, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate}));
}

void Q4Linear::addDecode(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &p, metal::MetalBuffer output, LinearMatrix matrix) const {
  add(graph, {input, output, {}, {}, {}, {}}, p, plan(decode(matrix, 1, LinearEpilogue::None)));
}

void Q4Linear::addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &p, metal::MetalBuffer output, LinearMatrix matrix,
    uint32_t lanes, Q4DispatchStats &stats) const {
  add(graph, {input, output, {}, {}, {}, {}}, p, plan(decode(matrix, lanes, LinearEpilogue::None)), nullptr, &stats);
}

void Q4Linear::addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &p, metal::MetalBuffer residual, metal::MetalBuffer output,
    LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats) const {
  add(graph, {input, output, {}, residual, {}, {}}, p, plan(decode(matrix, lanes, LinearEpilogue::Residual)), nullptr, &stats);
}

void Q4Linear::addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
    const Q8Projection &gate, const Q8Projection &up, metal::MetalBuffer gateScratch,
    metal::MetalBuffer output, LinearMatrix matrix, uint32_t lanes, Q4DispatchStats &stats) const {
  add(graph, {input, output, {}, {}, gateScratch, {}}, up, plan(decode(matrix, lanes, LinearEpilogue::GateUp)), &gate, &stats);
}

} // namespace splash::ops
