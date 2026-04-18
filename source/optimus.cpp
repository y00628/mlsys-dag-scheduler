#include "optimus.h"

#include "conv_accelerator.h"
#include "evaluator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlsys {

namespace {

struct OpGraph {
    std::vector<int> producer_of_tensor;
    std::vector<std::vector<size_t>> consumers_of_tensor;
    std::vector<std::vector<size_t>> preds;
    std::vector<std::vector<size_t>> succs;
    std::vector<size_t> topo_order;
    std::vector<size_t> topo_pos;
};

struct GroupBoundary {
    std::vector<size_t> boundary_inputs;
    std::vector<size_t> boundary_outputs;
    std::vector<size_t> internal_tensors;
};

struct GroupMetrics {
    bool valid = false;
    int64_t working_set_bytes = 0;
    double latency = std::numeric_limits<double>::infinity();
};

enum class LegalityLevel {
    kL0Graph = 0,
    kL1Execution,
    kL2Resource,
    kL3Policy,
};

enum class LegalityReason {
    kNone = 0,
    kDisconnectedSubgraph,
    kOrderingConflict,
    kBoundaryUnsatisfied,
    kWorkingSetOOM,
    kHeuristicRejectShape,
    kHeuristicRejectPolicy,
    kScorerRejected,
    kNoFeasibleGranularity,
    kPartitionCycle,
    kRuntimeBudgetExceeded,
};

struct LegalityResult {
    bool is_valid = false;
    LegalityLevel level_passed = LegalityLevel::kL0Graph;
    LegalityLevel failed_level = LegalityLevel::kL0Graph;
    LegalityReason reason = LegalityReason::kNone;
    int64_t estimated_working_set_bytes = 0;
    std::vector<size_t> violating_ops;
    std::vector<size_t> violating_tensors;
    std::string debug_note;
};

const char* ToString(LegalityLevel level) {
    switch (level) {
        case LegalityLevel::kL0Graph:
            return "L0_GRAPH";
        case LegalityLevel::kL1Execution:
            return "L1_EXECUTION";
        case LegalityLevel::kL2Resource:
            return "L2_RESOURCE";
        case LegalityLevel::kL3Policy:
            return "L3_POLICY";
    }
    return "UNKNOWN_LEVEL";
}

const char* ToString(LegalityReason reason) {
    switch (reason) {
        case LegalityReason::kNone:
            return "NONE";
        case LegalityReason::kDisconnectedSubgraph:
            return "DISCONNECTED_SUBGRAPH";
        case LegalityReason::kOrderingConflict:
            return "ORDERING_CONFLICT";
        case LegalityReason::kBoundaryUnsatisfied:
            return "BOUNDARY_UNSATISFIED";
        case LegalityReason::kWorkingSetOOM:
            return "WORKING_SET_OOM";
        case LegalityReason::kHeuristicRejectShape:
            return "HEURISTIC_REJECT_SHAPE";
        case LegalityReason::kHeuristicRejectPolicy:
            return "HEURISTIC_REJECT_POLICY";
        case LegalityReason::kScorerRejected:
            return "SCORER_REJECTED";
        case LegalityReason::kNoFeasibleGranularity:
            return "NO_FEASIBLE_GRANULARITY";
        case LegalityReason::kPartitionCycle:
            return "PARTITION_CYCLE";
        case LegalityReason::kRuntimeBudgetExceeded:
            return "RUNTIME_BUDGET_EXCEEDED";
    }
    return "UNKNOWN_REASON";
}

struct StepEstimate {
    int64_t active_bytes = 0;
    int64_t output_write_bytes = 0;
    double compute_cost = 0.0;
};

// Scheduler-driven analysis result stored per candidate.
struct SchedulerEstimate {
    bool valid = false;
    bool is_linear_chain = false;
    int64_t subgroup_count = 0;
    int64_t parameter_refills = 0;
    double estimated_memory_traffic_bytes = 0.0;
    double scheduler_latency_estimate = std::numeric_limits<double>::infinity();
    double confidence = 0.0;  // [0,1], lower for non-linear fallback
    // ISOA: populated for valid linear chains only.
    ISOADecision isoa;
    int64_t isoa_reduced_working_set_bytes = 0;
};

struct CandidateGroup {
    size_t start = 0;
    size_t end = 0;
    std::vector<size_t> ops;
    size_t seed_id = std::numeric_limits<size_t>::max();
    size_t growth_depth = 0;
    std::vector<size_t> topo_footprint;
    GroupBoundary boundary;
    Granularity granularity{};
    GroupMetrics metrics;
    LegalityResult legality;
    int64_t internalized_bytes = 0;
    SchedulerEstimate scheduler;  // Scheduler analysis (optional, default invalid)
};

enum class CandidateGenerationMode {
    kInterval = 0,
    kSeedGrowth,
};

enum class GuidanceMode {
    kContest = 0,
    kConvAccelerator,
};

enum class ConvGuidanceVariant {
    kAdditivePenaltyV1 = 0,
    kLocalRerankV2,
};

struct OptimusConfig {
    CandidateGenerationMode candidate_mode = CandidateGenerationMode::kInterval;
    GuidanceMode guidance_mode = GuidanceMode::kContest;
    ConvGuidanceVariant conv_guidance_variant =
        ConvGuidanceVariant::kAdditivePenaltyV1;
    // Scheduler integration settings
    double scheduler_cost_weight = 0.0;       // Blended cost: (1-w)*evaluator + w*scheduler
    bool scheduler_propose_granularity = false; // Inject scheduler-proposed granularities
    // When true, the seed-growth generator keeps non-topologically-contiguous
    // candidates and files them under the seed's topo-position (not
    // candidate.start). Required for the bitmask/frontier set-cover DP.
    bool allow_noncontig_groups = false;
};

struct ConvGuidanceMetrics {
    bool valid = false;
    double ranking_penalty = 0.0;
    int64_t working_set_bytes = 0;
    int64_t parameter_refills = 0;
    int64_t subgroup_count = 0;
    double memory_traffic_bytes = 0.0;
};

struct SearchDecision {
    bool valid = false;
    double cost = std::numeric_limits<double>::infinity();
    size_t candidate_index = 0;
    std::vector<size_t> retained_outputs;
};

struct SearchStateKey {
    size_t start = 0;
    std::vector<size_t> retained_inputs;

    bool operator==(const SearchStateKey& other) const {
        return start == other.start && retained_inputs == other.retained_inputs;
    }
};

struct SearchStateKeyHash {
    size_t operator()(const SearchStateKey& key) const {
        size_t h = std::hash<size_t>{}(key.start);
        for (size_t tensor_id : key.retained_inputs) {
            h ^= std::hash<size_t>{}(tensor_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct ScoreCacheKey {
    std::vector<size_t> ops;
    Granularity granularity{};
    std::vector<size_t> retained_inputs;
    std::vector<size_t> retained_outputs;

    bool operator==(const ScoreCacheKey& other) const {
        return ops == other.ops &&
               granularity.width == other.granularity.width &&
               granularity.height == other.granularity.height &&
               granularity.depth == other.granularity.depth &&
               retained_inputs == other.retained_inputs &&
               retained_outputs == other.retained_outputs;
    }
};

struct ScoreCacheKeyHash {
    size_t operator()(const ScoreCacheKey& key) const {
        size_t h = 0;
        auto mix = [&](size_t value) {
            h ^= value + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        for (size_t op_id : key.ops) {
            mix(std::hash<size_t>{}(op_id));
        }
        mix(std::hash<int64_t>{}(key.granularity.width));
        mix(std::hash<int64_t>{}(key.granularity.height));
        mix(std::hash<int64_t>{}(key.granularity.depth));
        for (size_t tensor_id : key.retained_inputs) {
            mix(std::hash<size_t>{}(tensor_id));
        }
        for (size_t tensor_id : key.retained_outputs) {
            mix(std::hash<size_t>{}(tensor_id));
        }
        return h;
    }
};

struct OpVectorHash {
    size_t operator()(const std::vector<size_t>& ops) const {
        size_t h = 0;
        for (size_t op_id : ops) {
            h ^= std::hash<size_t>{}(op_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr size_t kDefaultMaxGroupSize = 8;
constexpr size_t kGranularityRefineBudget = 6;
constexpr size_t kConvRerankBudget = 8;

struct SeedGrowthRuntimeConfig {
    size_t max_group_size_override = 0;
    size_t max_frontier = 128;
    size_t max_candidates_per_start = 8;
    size_t total_queue_budget = 50000;
    size_t max_states_per_seed = 0;
    bool allow_predecessor_growth = true;
    bool allow_successor_growth = true;
    bool require_contiguous_topo = true;
};

struct LegalityPolicyConfig {
    bool enable_l3_policy = true;
    bool enforce_common_output_shape = true;
    bool enforce_internalized_multi_op = true;
};

std::unordered_map<ScoreCacheKey, GroupMetrics, ScoreCacheKeyHash> g_score_cache;

int64_t CeilDivInt(int64_t value, int64_t divisor) {
    if (divisor <= 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

int64_t TensorElements(const Tensor& tensor) {
    return tensor.width * tensor.height;
}

int64_t TensorBytesForSet(const Problem& problem,
                         const std::vector<size_t>& tensors) {
    int64_t total = 0;
    for (size_t tensor_id : tensors) {
        total += TensorElements(problem.tensors[tensor_id]);
    }
    return total;
}

int64_t ReadInt64EnvOrDefault(const char* name, int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw) {
        return default_value;
    }
    return static_cast<int64_t>(parsed);
}

int64_t MatMulK(const Problem& problem, size_t op_id) {
    const auto& op = problem.ops[op_id];
    if (op.inputs.size() < 2) {
        return 1;
    }
    return std::max<int64_t>(1, problem.tensors[op.inputs[0]].width);
}

bool IsMatMul(const Op& op) {
    return op.op_type == "MatMul";
}

bool IsPointwise(const Op& op) {
    return op.op_type == "Pointwise";
}

bool ContainsMatMul(const Problem& problem, const std::vector<size_t>& ops) {
    for (size_t op_id : ops) {
        if (IsMatMul(problem.ops[op_id])) {
            return true;
        }
    }
    return false;
}

CandidateGenerationMode GetCandidateGenerationMode() {
    const char* raw = std::getenv("MLSYS_OPTIMUS_CANDIDATES");
    if (raw == nullptr) {
        return CandidateGenerationMode::kSeedGrowth;
    }

    std::string mode(raw);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "interval") {
        return CandidateGenerationMode::kInterval;
    }
    if (mode == "seed" || mode == "seed-growth" || mode == "seed_growth") {
        return CandidateGenerationMode::kSeedGrowth;
    }
    return CandidateGenerationMode::kSeedGrowth;
}

bool SeedDebugEnabled() {
    const char* raw = std::getenv("MLSYS_OPTIMUS_DEBUG_SEED");
    return raw != nullptr && std::string(raw) == "1";
}

bool SeedDebugVerboseEnabled() {
    const char* raw = std::getenv("MLSYS_OPTIMUS_DEBUG_SEED_VERBOSE");
    return raw != nullptr && std::string(raw) == "1";
}

bool LegalityDebugEnabled() {
    const char* raw = std::getenv("MLSYS_OPTIMUS_DEBUG_LEGALITY");
    return raw != nullptr && std::string(raw) == "1";
}

size_t ReadSizeTEnvOrDefault(const char* name, size_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

bool ReadBoolEnvOrDefault(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return default_value;
}

SeedGrowthRuntimeConfig GetSeedGrowthRuntimeConfig() {
    SeedGrowthRuntimeConfig config;
    config.max_group_size_override =
        ReadSizeTEnvOrDefault("MLSYS_OPTIMUS_SEED_MAX_GROUP", 0);
    config.max_frontier =
        std::max<size_t>(1, ReadSizeTEnvOrDefault("MLSYS_OPTIMUS_SEED_MAX_FRONTIER", 128));
    config.max_candidates_per_start = std::max<size_t>(
        1, ReadSizeTEnvOrDefault("MLSYS_OPTIMUS_SEED_MAX_CANDIDATES_PER_START", 8));
    config.total_queue_budget = std::max<size_t>(
        1, ReadSizeTEnvOrDefault("MLSYS_OPTIMUS_SEED_TOTAL_QUEUE_BUDGET", 50000));
    config.max_states_per_seed = ReadSizeTEnvOrDefault(
        "MLSYS_OPTIMUS_SEED_MAX_STATES_PER_SEED", 0);
    config.allow_predecessor_growth = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_SEED_ALLOW_PRED", true);
    config.allow_successor_growth = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_SEED_ALLOW_SUCC", true);
    config.require_contiguous_topo = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_SEED_REQUIRE_CONTIGUOUS", true);
    if (!config.allow_predecessor_growth && !config.allow_successor_growth) {
        config.allow_successor_growth = true;
    }
    return config;
}

LegalityPolicyConfig GetLegalityPolicyConfig() {
    LegalityPolicyConfig config;
    config.enable_l3_policy = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_LEGALITY_ENABLE_L3", true);
    config.enforce_common_output_shape = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_LEGALITY_ENFORCE_COMMON_OUTPUT_SHAPE", false);
    config.enforce_internalized_multi_op = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_LEGALITY_ENFORCE_INTERNALIZED_MULTI_OP", true);
    return config;
}

ConvDataflow ParseConvDataflow() {
    const char* raw = std::getenv("MLSYS_OPTIMUS_CONV_DATAFLOW");
    if (raw == nullptr) {
        return ConvDataflow::kRowStationary;
    }

    std::string mode(raw);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "os" || mode == "output" || mode == "output_stationary") {
        return ConvDataflow::kOutputStationary;
    }
    if (mode == "ws" || mode == "weight" || mode == "weight_stationary") {
        return ConvDataflow::kWeightStationary;
    }
    if (mode == "is" || mode == "input" || mode == "input_stationary") {
        return ConvDataflow::kInputStationary;
    }
    if (mode == "rs" || mode == "row" || mode == "row_stationary") {
        return ConvDataflow::kRowStationary;
    }
    return ConvDataflow::kRowStationary;
}

OpGraph BuildOpGraph(const Problem& problem) {
    OpGraph graph;
    graph.producer_of_tensor.assign(problem.tensors.size(), -1);
    graph.consumers_of_tensor.assign(problem.tensors.size(), {});

    for (size_t op_id = 0; op_id < problem.ops.size(); ++op_id) {
        for (size_t tensor_id : problem.ops[op_id].outputs) {
            graph.producer_of_tensor[tensor_id] = static_cast<int>(op_id);
        }
        for (size_t tensor_id : problem.ops[op_id].inputs) {
            graph.consumers_of_tensor[tensor_id].push_back(op_id);
        }
    }

    graph.preds.assign(problem.ops.size(), {});
    graph.succs.assign(problem.ops.size(), {});
    std::vector<int> indegree(problem.ops.size(), 0);
    for (size_t op_id = 0; op_id < problem.ops.size(); ++op_id) {
        for (size_t tensor_id : problem.ops[op_id].inputs) {
            int producer = graph.producer_of_tensor[tensor_id];
            if (producer >= 0) {
                graph.preds[op_id].push_back(static_cast<size_t>(producer));
                graph.succs[producer].push_back(op_id);
                indegree[op_id]++;
            }
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    while (!q.empty()) {
        size_t op_id = q.front();
        q.pop();
        graph.topo_order.push_back(op_id);
        for (size_t succ : graph.succs[op_id]) {
            if (--indegree[succ] == 0) {
                q.push(succ);
            }
        }
    }

    graph.topo_pos.assign(problem.ops.size(), 0);
    for (size_t i = 0; i < graph.topo_order.size(); ++i) {
        graph.topo_pos[graph.topo_order[i]] = i;
    }

    return graph;
}

std::vector<size_t> CanonicalizeOpSet(const std::vector<size_t>& ops,
                                      const OpGraph& graph) {
    std::vector<size_t> canonical = ops;
    std::sort(canonical.begin(), canonical.end(),
              [&](size_t lhs, size_t rhs) {
                  return graph.topo_pos[lhs] < graph.topo_pos[rhs];
              });
    canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());
    return canonical;
}

std::vector<size_t> CandidateKeyFromOps(const std::vector<size_t>& canonical_ops) {
    return canonical_ops;
}

bool SharesCommonOutputShape(const Problem& problem,
                             const std::vector<size_t>& ops) {
    if (ops.empty()) {
        return false;
    }
    const auto& first_out = problem.tensors[problem.ops[ops.front()].outputs.front()];
    for (size_t op_id : ops) {
        if (problem.ops[op_id].outputs.size() != 1) {
            return false;
        }
        const auto& out = problem.tensors[problem.ops[op_id].outputs.front()];
        if (out.width != first_out.width || out.height != first_out.height) {
            return false;
        }
    }
    return true;
}

bool IsConnectedSubDAG(const OpGraph& graph, const std::vector<size_t>& ops) {
    if (ops.empty()) {
        return false;
    }
    if (ops.size() == 1) {
        return true;
    }

    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    std::queue<size_t> q;
    std::unordered_set<size_t> visited;
    q.push(ops.front());
    visited.insert(ops.front());

    while (!q.empty()) {
        size_t op_id = q.front();
        q.pop();
        for (size_t next : graph.succs[op_id]) {
            if (op_set.count(next) && !visited.count(next)) {
                visited.insert(next);
                q.push(next);
            }
        }
        for (size_t prev : graph.preds[op_id]) {
            if (op_set.count(prev) && !visited.count(prev)) {
                visited.insert(prev);
                q.push(prev);
            }
        }
    }

    return visited.size() == ops.size();
}

bool IsLinearChainCandidate(const OpGraph& graph, const std::vector<size_t>& ops) {
    if (ops.empty()) {
        return false;
    }
    if (ops.size() == 1) {
        return true;
    }

    for (size_t i = 1; i < ops.size(); ++i) {
        bool linked = false;
        for (size_t pred : graph.preds[ops[i]]) {
            if (pred == ops[i - 1]) {
                linked = true;
                break;
            }
        }
        if (!linked) {
            return false;
        }
    }
    return true;
}

GroupBoundary ComputeBoundary(const Problem& problem, const OpGraph& graph,
                              const std::vector<size_t>& ops) {
    GroupBoundary boundary;
    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    std::unordered_set<size_t> boundary_inputs;
    std::unordered_set<size_t> boundary_outputs;
    std::unordered_set<size_t> internal_tensors;

    for (size_t op_id : ops) {
        for (size_t tensor_id : problem.ops[op_id].inputs) {
            int producer = graph.producer_of_tensor[tensor_id];
            if (producer < 0 || !op_set.count(static_cast<size_t>(producer))) {
                boundary_inputs.insert(tensor_id);
            } else {
                internal_tensors.insert(tensor_id);
            }
        }
        for (size_t tensor_id : problem.ops[op_id].outputs) {
            bool escapes = graph.consumers_of_tensor[tensor_id].empty();
            for (size_t consumer : graph.consumers_of_tensor[tensor_id]) {
                if (!op_set.count(consumer)) {
                    escapes = true;
                    break;
                }
            }
            if (escapes) {
                boundary_outputs.insert(tensor_id);
            }
        }
    }

    boundary.boundary_inputs.assign(boundary_inputs.begin(), boundary_inputs.end());
    boundary.boundary_outputs.assign(boundary_outputs.begin(), boundary_outputs.end());
    boundary.internal_tensors.assign(internal_tensors.begin(), internal_tensors.end());

    std::sort(boundary.boundary_inputs.begin(), boundary.boundary_inputs.end());
    std::sort(boundary.boundary_outputs.begin(), boundary.boundary_outputs.end());
    std::sort(boundary.internal_tensors.begin(), boundary.internal_tensors.end());

    return boundary;
}

ConvAcceleratorSpec BuildConvAcceleratorSpec(const Problem& problem) {
    ConvAcceleratorSpec spec;
    spec.on_chip_buffer_bytes = problem.fast_memory_capacity;
    spec.line_buffer_bytes = std::max<int64_t>(0, problem.fast_memory_capacity / 8);
    spec.pe_array_rows =
        std::max<int64_t>(1, std::min<int64_t>(problem.native_granularity.height, 64));
    spec.pe_array_cols =
        std::max<int64_t>(1, std::min<int64_t>(problem.native_granularity.width, 64));
    spec.rf_bytes_per_pe = 0;
    spec.pe_throughput = std::max<int64_t>(1, problem.native_granularity.depth);
    spec.dataflow = ParseConvDataflow();
    spec.supports_line_buffer = true;
    spec.supports_isoa = true;
    return spec;
}

Conv2DOp BuildPseudoConvOp(const Problem& problem, size_t op_id) {
    Conv2DOp conv;
    conv.name = problem.ops[op_id].op_type + "_" + std::to_string(op_id);
    conv.bytes_per_element = 1;

    const auto& op = problem.ops[op_id];
    const auto& out = problem.tensors[op.outputs.front()];
    conv.output_height = std::max<int64_t>(1, out.height);
    conv.output_width = 1;
    conv.output_channels = std::max<int64_t>(1, out.width);
    conv.kernel_height = 1;
    conv.kernel_width = 1;
    conv.stride_height = 1;
    conv.stride_width = 1;

    if (IsMatMul(op) && op.inputs.size() >= 2) {
        const auto& lhs = problem.tensors[op.inputs[0]];
        conv.input_height = std::max<int64_t>(1, lhs.height);
        conv.input_width = 1;
        conv.input_channels = std::max<int64_t>(1, lhs.width);
        return conv;
    }

    int64_t input_height = conv.output_height;
    int64_t input_channels = 0;
    for (size_t tensor_id : op.inputs) {
        const auto& tensor = problem.tensors[tensor_id];
        input_height = std::max<int64_t>(input_height, tensor.height);
        input_channels += std::max<int64_t>(1, tensor.width);
    }
    conv.input_height = std::max<int64_t>(1, input_height);
    conv.input_width = 1;
    conv.input_channels = std::max<int64_t>(1, input_channels);
    return conv;
}

ConvTileShape BuildPseudoConvTerminalTile(const Problem& problem,
                                          const std::vector<size_t>& ops,
                                          const Granularity& granularity) {
    ConvTileShape tile;
    if (ops.empty()) {
        return tile;
    }
    const auto& out = problem.tensors[problem.ops[ops.back()].outputs.front()];
    tile.output_height =
        std::max<int64_t>(1, std::min<int64_t>(granularity.height, out.height));
    tile.output_width = 1;
    tile.output_channels =
        std::max<int64_t>(1, std::min<int64_t>(granularity.width, out.width));
    return tile;
}

ConvGuidanceMetrics AnalyzeConvGuidance(const Problem& problem, const OpGraph& graph,
                                        const std::vector<size_t>& ops,
                                        const Granularity& granularity) {
    ConvGuidanceMetrics metrics;
    if (!IsLinearChainCandidate(graph, ops)) {
        return metrics;
    }

    std::vector<Conv2DOp> conv_ops;
    conv_ops.reserve(ops.size());
    for (size_t op_id : ops) {
        conv_ops.push_back(BuildPseudoConvOp(problem, op_id));
    }

    const ConvAcceleratorSpec spec = BuildConvAcceleratorSpec(problem);
    const ConvTileShape terminal_tile =
        BuildPseudoConvTerminalTile(problem, ops, granularity);
    const ConvGroupScheduleEstimate estimate =
        AnalyzeConvChain(conv_ops, spec, terminal_tile);
    if (!estimate.valid) {
        return metrics;
    }

    metrics.valid = true;
    metrics.working_set_bytes = estimate.working_set.total_bytes;
    metrics.parameter_refills = estimate.parameter_refills;
    metrics.subgroup_count = estimate.subgroup_count;
    metrics.memory_traffic_bytes = estimate.estimated_memory_traffic_bytes;
    metrics.ranking_penalty =
        estimate.estimated_memory_traffic_bytes /
            std::max<double>(1.0, static_cast<double>(problem.slow_memory_bandwidth)) +
        static_cast<double>(estimate.parameter_refills) * 0.25 +
        static_cast<double>(std::max<int64_t>(0, estimate.subgroup_count - 1)) * 8.0;
    return metrics;
}

// ---------------------------------------------------------------------------
// Scheduler-driven analysis and granularity proposal (optimus_sched backend)
// ---------------------------------------------------------------------------

// Forward declaration for EstimateProxyLatency (defined later, used as fallback).
double EstimateProxyLatency(const Problem& problem, const OpGraph& graph,
                            const std::vector<size_t>& ops,
                            const GroupBoundary& boundary,
                            const Granularity& granularity);

// Reuse-aware analytical latency model.  O(k_steps) complexity.
// Models LHS reuse across column tiles, RHS reuse across row tiles,
// and gates on capacity (falls back to proxy when reuse set too large).
struct ReuseAwareEstimate {
    bool reuse_possible = false;
    double latency = std::numeric_limits<double>::infinity();
    double effective_traffic_bytes = 0.0;
    int64_t reuse_set_bytes = 0;
};

ReuseAwareEstimate EstimateSchedulerLatency(
    const Problem& problem, const OpGraph& graph,
    const std::vector<size_t>& ops, const GroupBoundary& boundary,
    const Granularity& granularity) {

    ReuseAwareEstimate result;
    if (ops.empty()) {
        return result;
    }

    const auto& ref_out =
        problem.tensors[problem.ops[ops.front()].outputs.front()];
    const int64_t tiles_w = CeilDivInt(ref_out.width, granularity.width);
    const int64_t tiles_h = CeilDivInt(ref_out.height, granularity.height);
    const int64_t actual_h =
        std::min<int64_t>(granularity.height, ref_out.height);
    const int64_t actual_w =
        std::min<int64_t>(granularity.width, ref_out.width);

    int64_t max_k_steps = 1;
    for (size_t op_id : ops) {
        if (IsMatMul(problem.ops[op_id])) {
            max_k_steps = std::max(
                max_k_steps,
                CeilDivInt(MatMulK(problem, op_id), granularity.depth));
        }
    }

    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    std::unordered_set<size_t> boundary_outputs(
        boundary.boundary_outputs.begin(), boundary.boundary_outputs.end());

    // 1. Compute reuse set — worst case bytes simultaneously resident when
    //    processing one row of tiles (row-major traversal assumed).
    //    LHS stays for the whole row; RHS is reloaded per row but stays across
    //    columns within a row only if it fits.
    int64_t reuse_set_bytes = 0;
    for (size_t op_id : ops) {
        const auto& op = problem.ops[op_id];
        if (IsMatMul(op)) {
            // LHS tile for one k-step: h * k
            if (!op.inputs.empty()) {
                int p = graph.producer_of_tensor[op.inputs[0]];
                if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                    reuse_set_bytes += actual_h * granularity.depth;
                }
            }
            // RHS tile for one k-step: w * k  (per column tile)
            if (op.inputs.size() >= 2) {
                int p = graph.producer_of_tensor[op.inputs[1]];
                if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                    reuse_set_bytes += actual_w * granularity.depth;
                }
            }
            // Output accumulator tile: h * w
            reuse_set_bytes += actual_h * actual_w;
        } else if (IsPointwise(op)) {
            for (size_t tensor_id : op.inputs) {
                int p = graph.producer_of_tensor[tensor_id];
                if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                    reuse_set_bytes += actual_h * actual_w;
                }
            }
            if (boundary_outputs.count(op.outputs.front())) {
                reuse_set_bytes += actual_h * actual_w;
            }
        }
    }
    result.reuse_set_bytes = reuse_set_bytes;

    // 2. Capacity gate: if reuse set doesn't fit, fall back to proxy.
    if (reuse_set_bytes > problem.fast_memory_capacity) {
        result.reuse_possible = false;
        result.latency =
            EstimateProxyLatency(problem, graph, ops, boundary, granularity);
        result.effective_traffic_bytes =
            result.latency *
            static_cast<double>(problem.slow_memory_bandwidth);
        return result;
    }

    result.reuse_possible = true;

    // 3. Compute reuse-aware per-step latency.
    const int64_t native_w =
        std::max<int64_t>(1, problem.native_granularity.width);
    const int64_t native_h =
        std::max<int64_t>(1, problem.native_granularity.height);
    const int64_t native_k =
        std::max<int64_t>(1, problem.native_granularity.depth);
    const double spatial_factor = static_cast<double>(
        CeilDivInt(granularity.width, native_w) *
        CeilDivInt(granularity.height, native_h));
    const double bw =
        std::max<double>(1.0, static_cast<double>(problem.slow_memory_bandwidth));

    double total_latency = 0.0;
    double total_traffic = 0.0;

    for (int64_t step_index = 0; step_index < max_k_steps; ++step_index) {
        const bool final_k_step = (step_index == max_k_steps - 1);
        const int64_t k_begin = step_index * granularity.depth;

        double compute_cost = 0.0;
        double memory_bytes = 0.0;

        for (size_t op_id : ops) {
            const auto& op = problem.ops[op_id];

            if (IsMatMul(op)) {
                const int64_t k_total = MatMulK(problem, op_id);
                if (k_begin >= k_total) continue;

                const int64_t actual_k_step =
                    std::min<int64_t>(granularity.depth, k_total - k_begin);

                // Compute cost (same formula as EstimateStep).
                compute_cost +=
                    static_cast<double>(op.base_cost) * spatial_factor *
                    (static_cast<double>(actual_k_step) /
                     static_cast<double>(native_k));

                // LHS (activation): h × k_step, reused across column tiles.
                // Amortized: load once per row of tiles → divide by tiles_w.
                if (!op.inputs.empty()) {
                    int p = graph.producer_of_tensor[op.inputs[0]];
                    if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                        memory_bytes += static_cast<double>(
                                            actual_h * actual_k_step) /
                                        static_cast<double>(tiles_w);
                    }
                }
                // RHS (weight): w × k_step, reused across row tiles.
                // Amortized: load once per column of tiles → divide by tiles_h.
                if (op.inputs.size() >= 2) {
                    int p = graph.producer_of_tensor[op.inputs[1]];
                    if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                        memory_bytes += static_cast<double>(
                                            actual_w * actual_k_step) /
                                        static_cast<double>(tiles_h);
                    }
                }
                // Output: stays in fast memory across k-steps, written once.
                if (final_k_step &&
                    boundary_outputs.count(op.outputs.front())) {
                    memory_bytes += actual_h * actual_w;
                }
            } else if (IsPointwise(op) && final_k_step) {
                compute_cost +=
                    static_cast<double>(op.base_cost) * spatial_factor;

                // Pointwise boundary inputs: loaded per tile, no spatial reuse.
                for (size_t tensor_id : op.inputs) {
                    int p = graph.producer_of_tensor[tensor_id];
                    if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                        memory_bytes += actual_h * actual_w;
                    }
                }
                // Output.
                if (boundary_outputs.count(op.outputs.front())) {
                    memory_bytes += actual_h * actual_w;
                }
            }
        }

        const double memory_time = memory_bytes / bw;
        total_latency += std::max(compute_cost, memory_time);
        total_traffic += memory_bytes;
    }

    result.latency =
        total_latency * static_cast<double>(tiles_h * tiles_w);
    result.effective_traffic_bytes =
        total_traffic * static_cast<double>(tiles_h * tiles_w);
    return result;
}

SchedulerEstimate RunSchedulerAnalysis(const Problem& problem, const OpGraph& graph,
                                       const std::vector<size_t>& ops,
                                       const GroupBoundary& boundary,
                                       const Granularity& granularity) {
    SchedulerEstimate estimate;
    estimate.is_linear_chain = IsLinearChainCandidate(graph, ops);

    // Run the reuse-aware analytical latency model.
    const ReuseAwareEstimate reuse_est =
        EstimateSchedulerLatency(problem, graph, ops, boundary, granularity);

    if (estimate.is_linear_chain) {
        // Also run conv-accelerator analysis for subgroup/refill metadata.
        std::vector<Conv2DOp> conv_ops;
        conv_ops.reserve(ops.size());
        for (size_t op_id : ops) {
            conv_ops.push_back(BuildPseudoConvOp(problem, op_id));
        }

        const ConvAcceleratorSpec spec = BuildConvAcceleratorSpec(problem);
        const ConvTileShape terminal_tile =
            BuildPseudoConvTerminalTile(problem, ops, granularity);
        const ConvGroupScheduleEstimate conv_est =
            AnalyzeConvChain(conv_ops, spec, terminal_tile);

        if (conv_est.valid) {
            estimate.valid = true;
            estimate.subgroup_count = conv_est.subgroup_count;
            estimate.parameter_refills = conv_est.parameter_refills;
            estimate.estimated_memory_traffic_bytes =
                reuse_est.effective_traffic_bytes;
            estimate.scheduler_latency_estimate = reuse_est.latency;
            // Capture ISOA decision for downstream working-set correction.
            estimate.isoa = conv_est.isoa;
            estimate.isoa_reduced_working_set_bytes =
                conv_est.isoa.minimum_activation_bytes +
                conv_est.working_set.parameter_bytes +
                conv_est.working_set.line_buffer_bytes;

            // Confidence: higher when reuse is possible.
            double confidence = reuse_est.reuse_possible ? 0.9 : 0.5;
            if (conv_est.subgroup_count > 4) confidence *= 0.85;
            if (conv_est.subgroup_count > 16) confidence *= 0.75;
            if (conv_est.parameter_refills > static_cast<int64_t>(ops.size())) {
                confidence *= 0.9;
            }
            estimate.confidence = std::max(0.1, std::min(1.0, confidence));
        } else {
            // Conv analysis failed but we still have the reuse estimate.
            estimate.valid = true;
            estimate.subgroup_count = 1;
            estimate.parameter_refills = 1;
            estimate.estimated_memory_traffic_bytes =
                reuse_est.effective_traffic_bytes;
            estimate.scheduler_latency_estimate = reuse_est.latency;
            estimate.confidence = reuse_est.reuse_possible ? 0.6 : 0.3;
        }
    } else {
        // Non-linear groups: reuse-aware estimate with lower confidence.
        estimate.valid = true;
        estimate.estimated_memory_traffic_bytes =
            reuse_est.effective_traffic_bytes;
        estimate.scheduler_latency_estimate = reuse_est.latency;
        estimate.subgroup_count = 1;
        estimate.parameter_refills = 1;
        estimate.confidence = reuse_est.reuse_possible ? 0.4 : 0.15;
    }

    return estimate;
}

// Forward declarations needed by ProposeSchedulerGranularity.
int64_t EstimateWorkingSetBytes(const Problem& problem, const OpGraph& graph,
                                const std::vector<size_t>& ops,
                                const GroupBoundary& boundary,
                                const Granularity& granularity);
double EstimateProxyLatency(const Problem& problem, const OpGraph& graph,
                            const std::vector<size_t>& ops,
                            const GroupBoundary& boundary,
                            const Granularity& granularity);
std::vector<int64_t> BuildKCandidates(const Problem& problem,
                                       const std::vector<size_t>& ops);
struct TileBufferCoeffs {
    double area_coeff = 0.0;
    double row_coeff  = 0.0;
    double col_coeff  = 0.0;
};
TileBufferCoeffs ComputeBufferCoeffs(const Problem& problem,
                                      const OpGraph& graph,
                                      const std::vector<size_t>& ops,
                                      const GroupBoundary& boundary);
int64_t SolveQuadraticTileWidth(double A_r, double B, double C);

std::vector<Granularity> ProposeSchedulerGranularity(
    const Problem& problem, const OpGraph& graph,
    const std::vector<size_t>& ops, const GroupBoundary& boundary,
    size_t max_proposals = 4) {

    std::vector<Granularity> proposals;
    if (!IsLinearChainCandidate(graph, ops)) {
        return proposals;  // Only propose for linear chains.
    }

    // Build pseudo-conv representation.
    std::vector<Conv2DOp> conv_ops;
    conv_ops.reserve(ops.size());
    for (size_t op_id : ops) {
        conv_ops.push_back(BuildPseudoConvOp(problem, op_id));
    }
    const ConvAcceleratorSpec spec = BuildConvAcceleratorSpec(problem);

    const auto& ref_out =
        problem.tensors[problem.ops[ops.front()].outputs.front()];
    const int64_t out_h = std::max<int64_t>(1, ref_out.height);
    const int64_t out_w = std::max<int64_t>(1, ref_out.width);
    const double C = static_cast<double>(problem.fast_memory_capacity);

    const auto k_candidates = BuildKCandidates(problem, ops);

    // Aspect ratios to try: natural, square, inverse.
    std::vector<double> ratios;
    ratios.push_back(static_cast<double>(out_h) / static_cast<double>(out_w));
    if (out_h != out_w) {
        ratios.push_back(1.0);
    }
    ratios.push_back(static_cast<double>(out_w) / static_cast<double>(out_h));

    // Use the analytical quadratic solve to find max tile, then validate with
    // the scheduler (AnalyzeConvChain) rather than just the proxy.
    const TileBufferCoeffs coeffs =
        ComputeBufferCoeffs(problem, graph, ops, boundary);

    struct ProposalWithScore {
        Granularity g;
        double traffic;
    };
    std::vector<ProposalWithScore> scored;

    for (int64_t k_step : k_candidates) {
        if (k_step <= 0) continue;
        for (double r : ratios) {
            if (r <= 0.0) continue;

            const double A_r = coeffs.area_coeff * r;
            const double B = (coeffs.row_coeff * r + coeffs.col_coeff) *
                             static_cast<double>(k_step);
            int64_t max_tw = SolveQuadraticTileWidth(A_r, B, C);
            if (max_tw <= 0) continue;

            int64_t tile_w = std::min<int64_t>(out_w, max_tw);
            int64_t tile_h = std::min<int64_t>(
                out_h,
                static_cast<int64_t>(r * static_cast<double>(tile_w)));
            tile_h = std::max<int64_t>(1, tile_h);
            tile_w = std::max<int64_t>(1, tile_w);

            Granularity g{tile_w, tile_h, k_step};

            // Validate with scheduler (conv chain analysis).
            ConvTileShape terminal_tile =
                BuildPseudoConvTerminalTile(problem, ops, g);
            ConvGroupScheduleEstimate est =
                AnalyzeConvChain(conv_ops, spec, terminal_tile);
            if (!est.valid ||
                est.working_set.total_bytes > problem.fast_memory_capacity) {
                // Try halving.
                tile_w = std::max<int64_t>(1, tile_w / 2);
                tile_h = std::max<int64_t>(1, tile_h / 2);
                g = {tile_w, tile_h, k_step};
                terminal_tile = BuildPseudoConvTerminalTile(problem, ops, g);
                est = AnalyzeConvChain(conv_ops, spec, terminal_tile);
                if (!est.valid ||
                    est.working_set.total_bytes > problem.fast_memory_capacity) {
                    continue;
                }
            }

            scored.push_back({g, est.estimated_memory_traffic_bytes});
        }
    }

    // Sort by traffic (lower is better).
    std::sort(scored.begin(), scored.end(),
              [](const ProposalWithScore& a, const ProposalWithScore& b) {
                  return a.traffic < b.traffic;
              });

    // Deduplicate and take top max_proposals.
    std::set<std::tuple<int64_t, int64_t, int64_t>> seen;
    for (const auto& p : scored) {
        auto key = std::make_tuple(p.g.width, p.g.height, p.g.depth);
        if (seen.insert(key).second) {
            proposals.push_back(p.g);
            if (proposals.size() >= max_proposals) break;
        }
    }

    return proposals;
}

double ComputeBlendedCost(double evaluator_latency,
                          const SchedulerEstimate& scheduler,
                          double weight) {
    if (!scheduler.valid || weight <= 0.0) {
        return evaluator_latency;
    }
    const double effective_weight = weight * scheduler.confidence;
    return (1.0 - effective_weight) * evaluator_latency +
           effective_weight * scheduler.scheduler_latency_estimate;
}

bool ShouldApplyConvGuidanceV2(const Problem& problem, const OpGraph& graph,
                               const std::vector<size_t>& ops) {
    return ops.size() >= 2 && ContainsMatMul(problem, ops) &&
           IsLinearChainCandidate(graph, ops);
}

void ApplyConvLocalRerankV2(
    const Problem& problem, const OpGraph& graph,
    const std::vector<size_t>& ops,
    std::vector<std::pair<double, Granularity>>* proxy_ranked) {
    if (proxy_ranked == nullptr || proxy_ranked->empty()) {
        return;
    }
    if (!ShouldApplyConvGuidanceV2(problem, graph, ops)) {
        return;
    }

    const size_t rerank_budget = std::min(kConvRerankBudget, proxy_ranked->size());
    std::vector<std::pair<double, Granularity>> top_candidates(
        proxy_ranked->begin(), proxy_ranked->begin() + rerank_budget);

    struct RerankedCandidate {
        double score = kInfinity;
        double proxy_score = kInfinity;
        Granularity granularity{};
        bool pruned = false;
    };

    std::vector<RerankedCandidate> reranked;
    reranked.reserve(top_candidates.size());
    for (const auto& entry : top_candidates) {
        RerankedCandidate ranked;
        ranked.proxy_score = entry.first;
        ranked.granularity = entry.second;

        const ConvGuidanceMetrics conv_metrics =
            AnalyzeConvGuidance(problem, graph, ops, entry.second);
        if (!conv_metrics.valid) {
            ranked.score = ranked.proxy_score;
            reranked.push_back(ranked);
            continue;
        }
        if (conv_metrics.working_set_bytes > problem.fast_memory_capacity ||
            conv_metrics.parameter_refills >
                static_cast<int64_t>(2 * std::max<size_t>(1, ops.size())) ||
            conv_metrics.subgroup_count > 4) {
            ranked.pruned = true;
            reranked.push_back(ranked);
            continue;
        }

        const double conv_traffic_time =
            conv_metrics.memory_traffic_bytes /
            std::max<double>(1.0, static_cast<double>(problem.slow_memory_bandwidth));
        const double traffic_ratio =
            conv_traffic_time / std::max<double>(1.0, ranked.proxy_score);
        const double normalized_penalty = 0.15 * traffic_ratio;
        ranked.score = ranked.proxy_score * (1.0 + normalized_penalty);
        reranked.push_back(ranked);
    }

    std::stable_sort(reranked.begin(), reranked.end(),
                     [](const RerankedCandidate& lhs,
                        const RerankedCandidate& rhs) {
                         if (lhs.pruned != rhs.pruned) {
                             return !lhs.pruned;
                         }
                         if (lhs.score != rhs.score) {
                             return lhs.score < rhs.score;
                         }
                         return lhs.proxy_score < rhs.proxy_score;
                     });

    size_t out_index = 0;
    for (const auto& ranked : reranked) {
        if (!ranked.pruned) {
            (*proxy_ranked)[out_index++] = {ranked.proxy_score, ranked.granularity};
        }
    }
    for (const auto& ranked : reranked) {
        if (ranked.pruned) {
            (*proxy_ranked)[out_index++] = {ranked.proxy_score, ranked.granularity};
        }
    }
}

std::vector<int64_t> BuildSpatialCandidates(int64_t dim, int64_t native_dim) {
    std::set<int64_t> values;
    const int64_t safe_dim = std::max<int64_t>(1, dim);
    const int64_t safe_native = std::max<int64_t>(1, native_dim);

    values.insert(safe_dim);

    for (int64_t divisor : {1LL, 2LL, 4LL, 8LL, 16LL}) {
        values.insert(std::max<int64_t>(1, std::min(safe_dim, safe_native / divisor))); 
    }
    for (int64_t tiles : {1LL, 2LL, 4LL, 8LL, 16LL, 32LL}) {
        values.insert(std::max<int64_t>(1, CeilDivInt(safe_dim, tiles)));
    }
    // Small multiples of native granularity (2x through 8x).
    // The existing loops above only produce native/power-of-2 and ceil(dim/power-of-2),
    // which misses hardware-aligned intermediate sizes. For example, with native=128
    // and dim=2048, the loops above give {8,16,32,64,128,256,512,1024,2048} but miss
    // 384 (3×128), 640 (5×128), 768 (6×128), 896 (7×128). These intermediate sizes
    // can be optimal when capacity constraints are tight — they give more tiles than
    // 256 but fewer than 128, hitting a sweet spot between parallelism and memory.
    for (int64_t mult = 2; mult <= 8; ++mult) {
        const int64_t v = mult * safe_native;
        if (v <= safe_dim) {
            values.insert(v);
        }
    }
    // Divisor-friendly multiples: values where dim % v == 0, meaning every tile is
    // exactly the same size (no smaller "edge" tile at the boundary). This eliminates
    // edge tile waste entirely and gives the most uniform workload distribution.
    // E.g., for dim=2048, native=128: 384 doesn't divide 2048, but 256 (2×128) and
    // 512 (4×128) do, so those get added here (if not already present).
    for (int64_t mult : {2LL, 3LL, 4LL, 5LL, 6LL, 7LL, 8LL, 10LL, 12LL, 16LL}) {
        const int64_t v = mult * safe_native;
        if (v > 0 && v <= safe_dim && safe_dim % v == 0) {
            values.insert(v);
        }
    }

    std::vector<int64_t> result(values.begin(), values.end());
    std::sort(result.begin(), result.end(), std::greater<int64_t>());
    return result;
}

std::vector<int64_t> BuildKCandidates(const Problem& problem,
                                      const std::vector<size_t>& ops) {
    bool has_matmul = false;
    int64_t max_k = 1;
    for (size_t op_id : ops) {
        if (IsMatMul(problem.ops[op_id])) {
            has_matmul = true;
            max_k = std::max(max_k, MatMulK(problem, op_id));
        }
    }
    if (!has_matmul) {
        return {1};
    }

    std::set<int64_t> values;
    const int64_t native_k = std::max<int64_t>(1, problem.native_granularity.depth);
    values.insert(max_k);
    for (int64_t divisor : {1LL, 2LL, 4LL, 8LL, 16LL}) {
        values.insert(std::max<int64_t>(1, std::min(max_k, native_k / divisor)));
    }
    for (int64_t steps : {1LL, 2LL, 4LL, 8LL, 16LL}) {
        values.insert(std::max<int64_t>(1, CeilDivInt(max_k, steps)));
    }

    std::vector<int64_t> result(values.begin(), values.end());
    std::sort(result.begin(), result.end(), std::greater<int64_t>());
    return result;
}

int64_t EstimateWorkingSetBytes(const Problem& problem, const OpGraph& graph,
                                const std::vector<size_t>& ops,
                                const GroupBoundary& boundary,
                                const Granularity& granularity);

double EstimateProxyLatency(const Problem& problem, const OpGraph& graph,
                            const std::vector<size_t>& ops,
                            const GroupBoundary& boundary,
                            const Granularity& granularity);

void AddAnalyticalTileCandidates(const Problem& problem,
                                  const OpGraph& graph,
                                  const std::vector<size_t>& ops,
                                  const GroupBoundary& boundary,
                                  const std::vector<int64_t>& k_steps,
                                  std::vector<std::pair<double, Granularity>>* proxy_ranked);

StepEstimate EstimateStep(const Problem& problem, const OpGraph& graph,
                          const std::vector<size_t>& ops,
                          const GroupBoundary& boundary,
                          const Granularity& granularity, int64_t actual_h,
                          int64_t actual_w, int64_t step_index,
                          int64_t max_k_steps) {
    StepEstimate step;
    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    std::unordered_set<size_t> boundary_outputs(boundary.boundary_outputs.begin(),
                                                boundary.boundary_outputs.end());
    const int64_t native_w = std::max<int64_t>(1, problem.native_granularity.width);
    const int64_t native_h = std::max<int64_t>(1, problem.native_granularity.height);
    const int64_t native_k = std::max<int64_t>(1, problem.native_granularity.depth);
    const double spatial_factor =
        static_cast<double>(CeilDivInt(granularity.width, native_w) *
                            CeilDivInt(granularity.height, native_h));
    const bool final_k_step = (step_index == max_k_steps - 1);
    const int64_t k_begin = step_index * granularity.depth;

    for (size_t op_id : ops) {
        const auto& op = problem.ops[op_id];
        if (IsPointwise(op)) {
            if (!final_k_step) {
                continue;
            }
            step.compute_cost += static_cast<double>(op.base_cost) * spatial_factor;
            for (size_t tensor_id : op.inputs) {
                int producer = graph.producer_of_tensor[tensor_id];
                if (producer >= 0 && op_set.count(static_cast<size_t>(producer))) {
                    continue;
                }
                step.active_bytes += actual_h * actual_w;
            }
            if (boundary_outputs.count(op.outputs.front())) {
                step.active_bytes += actual_h * actual_w;
                step.output_write_bytes += actual_h * actual_w;
            }
            continue;
        }

        if (!IsMatMul(op)) {
            continue;
        }

        const int64_t k_total = MatMulK(problem, op_id);
        if (k_begin >= k_total) {
            continue;
        }
        const int64_t k_step =
            std::min<int64_t>(granularity.depth, k_total - k_begin);
        step.compute_cost += static_cast<double>(op.base_cost) * spatial_factor *
                             (static_cast<double>(k_step) /
                              static_cast<double>(native_k));

        for (size_t input_index = 0; input_index < op.inputs.size(); ++input_index) {
            size_t tensor_id = op.inputs[input_index];
            int producer = graph.producer_of_tensor[tensor_id];
            if (producer >= 0 && op_set.count(static_cast<size_t>(producer))) {
                continue;
            }
            if (input_index == 0) {
                step.active_bytes += actual_h * k_step;
            } else {
                step.active_bytes += actual_w * k_step;
            }
        }
        if (boundary_outputs.count(op.outputs.front())) {
            step.active_bytes += actual_h * actual_w;
            if (final_k_step) {
                step.output_write_bytes += actual_h * actual_w;
            }
        }
    }

    return step;
}

int64_t EstimateWorkingSetBytes(const Problem& problem, const OpGraph& graph,
                                const std::vector<size_t>& ops,
                                const GroupBoundary& boundary,
                                const Granularity& granularity) {
    if (ops.empty()) {
        return 0;
    }

    const auto& ref_out =
        problem.tensors[problem.ops[ops.front()].outputs.front()];

    int64_t max_k_steps = 1;
    for (size_t op_id : ops) {
        if (IsMatMul(problem.ops[op_id])) {
            max_k_steps = std::max(max_k_steps,
                                   CeilDivInt(MatMulK(problem, op_id),
                                              granularity.depth));
        }
    }

    // Peak active bytes is maximized at the interior tile (largest actual_h/actual_w).
    // Edge tiles have smaller dimensions and thus smaller working sets. O(k_steps) only.
    const int64_t actual_h = std::min<int64_t>(granularity.height, ref_out.height);
    const int64_t actual_w = std::min<int64_t>(granularity.width, ref_out.width);
    int64_t peak = 0;
    for (int64_t step_index = 0; step_index < max_k_steps; ++step_index) {
        peak = std::max(peak, EstimateStep(problem, graph, ops, boundary,
                                           granularity, actual_h, actual_w,
                                           step_index, max_k_steps)
                                  .active_bytes);
    }

    return peak;
}

double EstimateProxyLatency(const Problem& problem, const OpGraph& graph,
                            const std::vector<size_t>& ops,
                            const GroupBoundary& boundary,
                            const Granularity& granularity) {
    if (ops.empty()) {
        return kInfinity;
    }

    const auto& ref_out =
        problem.tensors[problem.ops[ops.front()].outputs.front()];
    const int64_t tiles_w = CeilDivInt(ref_out.width, granularity.width);
    const int64_t tiles_h = CeilDivInt(ref_out.height, granularity.height);

    int64_t max_k_steps = 1;
    for (size_t op_id : ops) {
        if (IsMatMul(problem.ops[op_id])) {
            max_k_steps = std::max(max_k_steps,
                                   CeilDivInt(MatMulK(problem, op_id),
                                              granularity.depth));
        }
    }

    // Edge-aware proxy: instead of assuming all tiles are the same size (interior),
    // we separate tiles into up to 4 categories based on position:
    //
    //   +------------------+-----+
    //   |                  |     |
    //   |   interior tiles | right-edge tiles
    //   |   (full_h×full_w)|     | (full_h × rem_w)
    //   |                  |     |
    //   +------------------+-----+
    //   | bottom-edge tiles|corner|
    //   | (rem_h × full_w) |(rem_h × rem_w)
    //   +------------------+-----+
    //
    // When granularity doesn't evenly divide the output tensor, the last row/column
    // of tiles are smaller. The old proxy multiplied interior cost by total tile count,
    // which overestimates latency for non-divisor tile sizes and biases against them.
    // This 4-category model is still O(4 × k_steps) — much faster than the evaluator's
    // O(tiles_h × tiles_w × k_steps) full simulation.
    const int64_t full_h = std::min<int64_t>(granularity.height, ref_out.height);
    const int64_t full_w = std::min<int64_t>(granularity.width, ref_out.width);
    const int64_t rem_h = ref_out.height % granularity.height;
    const int64_t rem_w = ref_out.width % granularity.width;
    const bool has_edge_h = (rem_h != 0 && tiles_h > 1);
    const bool has_edge_w = (rem_w != 0 && tiles_w > 1);

    // Compute total latency for one tile with given actual dimensions (ah × aw).
    // Sums roofline cost max(compute, memory) across all k-steps.
    auto compute_tile_latency = [&](int64_t ah, int64_t aw) -> double {
        double lat = 0.0;
        for (int64_t step = 0; step < max_k_steps; ++step) {
            const StepEstimate est = EstimateStep(
                problem, graph, ops, boundary, granularity, ah, aw,
                step, max_k_steps);
            const double mem =
                (static_cast<double>(est.active_bytes) +
                 static_cast<double>(est.output_write_bytes)) /
                static_cast<double>(problem.slow_memory_bandwidth);
            lat += std::max(est.compute_cost, mem);
        }
        return lat;
    };

    // Count tiles in each category and sum their costs.
    const int64_t interior_h = tiles_h - (has_edge_h ? 1 : 0);
    const int64_t interior_w = tiles_w - (has_edge_w ? 1 : 0);
    double total = compute_tile_latency(full_h, full_w) *
                   static_cast<double>(interior_h * interior_w);
    if (has_edge_w) {
        // Right-edge column: full height but reduced width (rem_w).
        total += compute_tile_latency(full_h, rem_w) *
                 static_cast<double>(interior_h);
    }
    if (has_edge_h) {
        // Bottom-edge row: reduced height (rem_h) but full width.
        total += compute_tile_latency(rem_h, full_w) *
                 static_cast<double>(interior_w);
    }
    if (has_edge_h && has_edge_w) {
        // Corner tile: reduced in both dimensions. At most 1.
        total += compute_tile_latency(rem_h, rem_w);
    }

    return total;
}

bool EvaluateWithOfficialScorer(const Problem& problem,
                                const std::vector<size_t>& ops,
                                const Granularity& granularity,
                                const std::vector<size_t>& retained_inputs,
                                const std::vector<size_t>& retained_outputs,
                                GroupMetrics* metrics) {
    if (metrics == nullptr) {
        return false;
    }

    ScoreCacheKey key{ops, granularity, retained_inputs, retained_outputs};
    std::sort(key.retained_inputs.begin(), key.retained_inputs.end());
    key.retained_inputs.erase(
        std::unique(key.retained_inputs.begin(), key.retained_inputs.end()),
        key.retained_inputs.end());
    std::sort(key.retained_outputs.begin(), key.retained_outputs.end());
    key.retained_outputs.erase(
        std::unique(key.retained_outputs.begin(), key.retained_outputs.end()),
        key.retained_outputs.end());

    auto cached = g_score_cache.find(key);
    if (cached != g_score_cache.end()) {
        *metrics = cached->second;
        return metrics->valid;
    }

    Subgraph subgraph;
    subgraph.op_ids = ops;
    subgraph.granularity = granularity;
    subgraph.tensors_to_retain = key.retained_outputs;
    subgraph.traversal_order = std::nullopt;
    subgraph.subgraph_latency = 0.0;

    SubgraphEvaluation evaluation;
    if (!EvaluateSubgraph(problem, subgraph, key.retained_inputs, &evaluation) ||
        !evaluation.valid) {
        g_score_cache[key] = *metrics;
        return false;
    }

    metrics->valid = true;
    metrics->latency = evaluation.latency;
    metrics->working_set_bytes = evaluation.peak_live_bytes;
    g_score_cache[key] = *metrics;
    return true;
}

GroupMetrics EvaluateGroup(const Problem& problem, const OpGraph& graph,
                           const std::vector<size_t>& ops,
                           const GroupBoundary& boundary,
                           const Granularity& granularity,
                           LegalityResult* legality_out) {
    GroupMetrics metrics;
    metrics.working_set_bytes =
        EstimateWorkingSetBytes(problem, graph, ops, boundary, granularity);
    if (legality_out != nullptr) {
        legality_out->is_valid = false;
        legality_out->level_passed = LegalityLevel::kL1Execution;
        legality_out->failed_level = LegalityLevel::kL2Resource;
        legality_out->reason = LegalityReason::kWorkingSetOOM;
        legality_out->estimated_working_set_bytes = metrics.working_set_bytes;
    }
    if (metrics.working_set_bytes > problem.fast_memory_capacity) {
        return metrics;
    }

    GroupMetrics aligned_metrics;
    if (!EvaluateWithOfficialScorer(problem, ops, granularity, {}, {},
                                    &aligned_metrics)) {
        if (legality_out != nullptr) {
            legality_out->is_valid = false;
            legality_out->level_passed = LegalityLevel::kL0Graph;
            legality_out->failed_level = LegalityLevel::kL1Execution;
            legality_out->reason = LegalityReason::kScorerRejected;
            legality_out->estimated_working_set_bytes = metrics.working_set_bytes;
        }
        return metrics;
    }
    if (legality_out != nullptr) {
        legality_out->is_valid = true;
        legality_out->level_passed = LegalityLevel::kL2Resource;
        legality_out->failed_level = LegalityLevel::kL3Policy;
        legality_out->reason = LegalityReason::kNone;
        legality_out->estimated_working_set_bytes = aligned_metrics.working_set_bytes;
        legality_out->debug_note = "valid";
    }
    aligned_metrics.working_set_bytes =
        std::max(aligned_metrics.working_set_bytes, metrics.working_set_bytes);
    return aligned_metrics;
}

bool RunStaticLegalityChecks(const Problem& problem, const OpGraph& graph,
                             CandidateGroup* candidate,
                             const LegalityPolicyConfig& policy) {
    if (candidate == nullptr) {
        return false;
    }
    if (candidate->ops.empty()) {
        candidate->legality.is_valid = false;
        candidate->legality.level_passed = LegalityLevel::kL0Graph;
        candidate->legality.failed_level = LegalityLevel::kL0Graph;
        candidate->legality.reason = LegalityReason::kOrderingConflict;
        candidate->legality.debug_note = "empty_candidate";
        return false;
    }

    if (!IsConnectedSubDAG(graph, candidate->ops)) {
        candidate->legality.is_valid = false;
        candidate->legality.level_passed = LegalityLevel::kL0Graph;
        candidate->legality.failed_level = LegalityLevel::kL0Graph;
        candidate->legality.reason = LegalityReason::kDisconnectedSubgraph;
        candidate->legality.violating_ops = candidate->ops;
        candidate->legality.debug_note = "disconnected_subdag";
        return false;
    }
    candidate->legality.level_passed = LegalityLevel::kL0Graph;

    if (candidate->boundary.boundary_outputs.empty()) {
        candidate->legality.is_valid = false;
        candidate->legality.level_passed = LegalityLevel::kL0Graph;
        candidate->legality.failed_level = LegalityLevel::kL1Execution;
        candidate->legality.reason = LegalityReason::kBoundaryUnsatisfied;
        candidate->legality.debug_note = "empty_boundary_outputs";
        return false;
    }
    candidate->legality.level_passed = LegalityLevel::kL1Execution;

    if (policy.enable_l3_policy) {
        if (policy.enforce_common_output_shape &&
            !SharesCommonOutputShape(problem, candidate->ops)) {
            candidate->legality.is_valid = false;
            candidate->legality.level_passed = LegalityLevel::kL1Execution;
            candidate->legality.failed_level = LegalityLevel::kL3Policy;
            candidate->legality.reason = LegalityReason::kHeuristicRejectShape;
            candidate->legality.debug_note = "shape_policy_reject";
            return false;
        }
        if (policy.enforce_internalized_multi_op && candidate->ops.size() > 1 &&
            candidate->boundary.internal_tensors.empty()) {
            candidate->legality.is_valid = false;
            candidate->legality.level_passed = LegalityLevel::kL1Execution;
            candidate->legality.failed_level = LegalityLevel::kL3Policy;
            candidate->legality.reason = LegalityReason::kHeuristicRejectPolicy;
            candidate->legality.debug_note = "no_internalized_tensor";
            return false;
        }
    }

    candidate->legality.level_passed = LegalityLevel::kL3Policy;
    return true;
}

// Decide how many proxy-ranked candidates to evaluate with the expensive official
// scorer. The proxy (fast, O(k_steps)) ranks ~450 candidates, but the evaluator
// (slow, O(tiles×k_steps) with LRU simulation) can only afford to score a few.
//
// The old fixed budget of 6 means only the top 1.3% get scored. If the proxy
// disagrees with the evaluator on ranking (common for multi-op fusion groups with
// complex memory patterns), the true best tile might be ranked 8th or 12th by
// proxy and never evaluated.
//
// This function adapts the budget based on two signals:
//   1. Group complexity: more ops = more proxy-evaluator divergence, so score more.
//   2. Score clustering: if many candidates have similar proxy scores (within 10%
//      of the best), the proxy can't distinguish them reliably, so let the
//      evaluator break the tie.
//
// Returns a budget in [6, 20], capped by the number of available candidates.
size_t ComputeRefineBudget(
    const std::vector<std::pair<double, Granularity>>& proxy_ranked,
    size_t num_ops) {
    constexpr size_t kMinBudget = 6;
    constexpr size_t kMaxBudget = 20;

    if (proxy_ranked.size() <= kMinBudget) {
        return proxy_ranked.size();
    }

    // Base budget scales with group complexity.
    size_t budget = kMinBudget;
    if (num_ops >= 2) budget += 2;  // 2-3 ops: budget = 8
    if (num_ops >= 4) budget += 2;  // 4+ ops:  budget = 10

    // Expand budget to cover the "ambiguous zone" where proxy scores are close.
    // proxy_ranked is already sorted ascending, so [0] is the best proxy score.
    if (proxy_ranked.size() >= 2) {
        const double threshold = proxy_ranked[0].first * 1.10;  // 10% window
        size_t similar = 0;
        for (const auto& [score, _] : proxy_ranked) {
            if (score <= threshold) {
                ++similar;
            } else {
                break;  // sorted, so all remaining are worse
            }
        }
        budget = std::max(budget, std::min(similar, kMaxBudget));
    }

    return std::min(budget, proxy_ranked.size());
}

// Returns the minimum working set bytes for a candidate group + granularity,
// taking ISOA into account for linear MatMul chains.
//
// ISOA only overrides the base estimate when the base estimate EXCEEDS the
// fast_memory_capacity threshold — i.e., ISOA is used as a rescue path to
// admit granularities that would otherwise be rejected, not as a general
// replacement for the proxy working-set estimate.  This avoids mis-ranking
// granularities that fit without ISOA by injecting a lower (but evaluator-
// misaligned) estimate for already-passing tiles.
//
// Returns the ISOA-reduced bytes only when base_ws > capacity AND isoa_ws fits.
// Otherwise returns base_ws unchanged.
int64_t ISOAAwareWorkingSetBytes(const Problem& problem, const OpGraph& graph,
                                  const std::vector<size_t>& ops,
                                  const GroupBoundary& boundary,
                                  const Granularity& granularity,
                                  bool is_linear_chain) {
    const int64_t base_ws =
        EstimateWorkingSetBytes(problem, graph, ops, boundary, granularity);
    // Fast path: already fits, or not a linear MatMul chain.
    if (base_ws <= problem.fast_memory_capacity ||
        !is_linear_chain || !ContainsMatMul(problem, ops)) {
        return base_ws;
    }
    // Base estimate exceeds capacity — try ISOA rescue.
    std::vector<Conv2DOp> conv_ops;
    conv_ops.reserve(ops.size());
    for (size_t op_id : ops) conv_ops.push_back(BuildPseudoConvOp(problem, op_id));
    const ConvAcceleratorSpec spec = BuildConvAcceleratorSpec(problem);
    const ConvTileShape terminal_tile =
        BuildPseudoConvTerminalTile(problem, ops, granularity);
    const ConvGroupScheduleEstimate conv_est =
        AnalyzeConvChain(conv_ops, spec, terminal_tile);
    if (!conv_est.valid || conv_est.isoa.minimum_activation_bytes <= 0) {
        return base_ws;
    }
    const int64_t isoa_ws = conv_est.isoa.minimum_activation_bytes +
                             conv_est.working_set.parameter_bytes +
                             conv_est.working_set.line_buffer_bytes;
    // Only use ISOA estimate if it's strictly better (rescues the candidate).
    return (isoa_ws < base_ws) ? isoa_ws : base_ws;
}

CandidateGroup BuildBestCandidate(const Problem& problem, const OpGraph& graph,
                                  const std::vector<size_t>& input_ops,
                                  const OptimusConfig& config) {
    CandidateGroup candidate;
    const LegalityPolicyConfig legality_policy = GetLegalityPolicyConfig();
    candidate.ops = input_ops;
    candidate.seed_id = input_ops.empty() ? std::numeric_limits<size_t>::max()
                                          : input_ops.front();
    candidate.growth_depth = input_ops.size();
    std::sort(candidate.ops.begin(), candidate.ops.end(),
              [&](size_t lhs, size_t rhs) {
                  return graph.topo_pos[lhs] < graph.topo_pos[rhs];
              });
    if (candidate.ops.empty()) {
        candidate.legality.is_valid = false;
        candidate.legality.level_passed = LegalityLevel::kL0Graph;
        candidate.legality.failed_level = LegalityLevel::kL0Graph;
        candidate.legality.reason = LegalityReason::kOrderingConflict;
        return candidate;
    }
    candidate.start = graph.topo_pos[candidate.ops.front()];
    candidate.end = graph.topo_pos[candidate.ops.back()];
    candidate.topo_footprint.reserve(candidate.ops.size());
    for (size_t op_id : candidate.ops) {
        candidate.topo_footprint.push_back(graph.topo_pos[op_id]);
    }

    candidate.boundary = ComputeBoundary(problem, graph, candidate.ops);
    for (size_t tensor_id : candidate.boundary.internal_tensors) {
        candidate.internalized_bytes += TensorElements(problem.tensors[tensor_id]);
    }

    if (!RunStaticLegalityChecks(problem, graph, &candidate, legality_policy)) {
        return candidate;
    }

    const auto& ref_out =
        problem.tensors[problem.ops[candidate.ops.front()].outputs.front()];
    const auto width_candidates =
        BuildSpatialCandidates(ref_out.width, problem.native_granularity.width);
    const auto height_candidates =
        BuildSpatialCandidates(ref_out.height, problem.native_granularity.height);
    const auto k_candidates = BuildKCandidates(problem, candidate.ops);
    std::vector<std::pair<double, Granularity>> proxy_ranked;
    const bool is_linear = IsLinearChainCandidate(graph, candidate.ops);

    for (int64_t width : width_candidates) {
        for (int64_t height : height_candidates) {
            for (int64_t depth : k_candidates) {
                Granularity granularity{width, height, depth};
                const int64_t working_set = ISOAAwareWorkingSetBytes(
                    problem, graph, candidate.ops, candidate.boundary,
                    granularity, is_linear);
                if (working_set > problem.fast_memory_capacity) {
                    continue;
                }
                double proxy_score = EstimateProxyLatency(
                    problem, graph, candidate.ops, candidate.boundary, granularity);
                if (config.guidance_mode == GuidanceMode::kConvAccelerator) {
                    if (config.conv_guidance_variant ==
                        ConvGuidanceVariant::kAdditivePenaltyV1) {
                        const ConvGuidanceMetrics conv_metrics =
                            AnalyzeConvGuidance(problem, graph, candidate.ops, granularity);
                        if (conv_metrics.valid) {
                            if (conv_metrics.working_set_bytes >
                                problem.fast_memory_capacity) {
                                continue;
                            }
                            proxy_score += 0.35 * conv_metrics.ranking_penalty;
                        }
                    }
                }
                proxy_ranked.push_back({proxy_score, granularity});
            }
        }
    }

    // Add analytically-derived max-tile candidates (Paper Eq. 4, ISOA).
    // For each k_step already in proxy_ranked, solve for the largest tile that
    // fits within fast_memory_capacity, accounting for parameter refill cost.
    {
        std::set<int64_t> tried_k;
        for (auto& [score, gran] : proxy_ranked) {
            tried_k.insert(gran.depth);
        }
        std::vector<int64_t> k_steps_to_try(tried_k.begin(), tried_k.end());
        if (k_steps_to_try.empty()) {
            k_steps_to_try = BuildKCandidates(problem, candidate.ops);
        }
        AddAnalyticalTileCandidates(problem, graph, candidate.ops,
                                     candidate.boundary, k_steps_to_try,
                                     &proxy_ranked);
    }

    std::sort(proxy_ranked.begin(), proxy_ranked.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    if (config.guidance_mode == GuidanceMode::kConvAccelerator &&
        config.conv_guidance_variant == ConvGuidanceVariant::kLocalRerankV2) {
        ApplyConvLocalRerankV2(problem, graph, candidate.ops, &proxy_ranked);
    }
    if (proxy_ranked.empty()) {
        candidate.legality.is_valid = false;
        candidate.legality.level_passed = LegalityLevel::kL1Execution;
        candidate.legality.failed_level = LegalityLevel::kL2Resource;
        candidate.legality.reason = LegalityReason::kNoFeasibleGranularity;
        candidate.legality.debug_note = "no_proxy_granularity";
        return candidate;
    }

    const size_t budget = ComputeRefineBudget(proxy_ranked, candidate.ops.size());
    for (size_t i = 0; i < budget; ++i) {
        GroupMetrics metrics = EvaluateGroup(problem, graph, candidate.ops,
                                             candidate.boundary,
                                             proxy_ranked[i].second,
                                             &candidate.legality);
        if (!metrics.valid) {
            continue;
        }
        if (!candidate.metrics.valid || metrics.latency < candidate.metrics.latency) {
            candidate.granularity = proxy_ranked[i].second;
            candidate.metrics = metrics;
            candidate.legality.is_valid = true;
            candidate.legality.level_passed = LegalityLevel::kL3Policy;
            candidate.legality.reason = LegalityReason::kNone;
            candidate.legality.failed_level = LegalityLevel::kL0Graph;
        }
    }

    if (!candidate.metrics.valid) {
        candidate.legality.is_valid = false;
        if (candidate.legality.reason == LegalityReason::kNone) {
            candidate.legality.level_passed = LegalityLevel::kL1Execution;
            candidate.legality.failed_level = LegalityLevel::kL2Resource;
            candidate.legality.reason = LegalityReason::kNoFeasibleGranularity;
            candidate.legality.debug_note = "granularity_refine_exhausted";
        }
    }

    // ISOA correction: if the evaluator-computed working set exceeds capacity,
    // try the ISOA-reduced estimate to allow the candidate to survive retain
    // choices in DP (IsCapacityFeasible uses candidate.metrics.working_set_bytes).
    if (candidate.metrics.valid && is_linear &&
        candidate.metrics.working_set_bytes > problem.fast_memory_capacity) {
        const int64_t isoa_ws = ISOAAwareWorkingSetBytes(
            problem, graph, candidate.ops, candidate.boundary,
            candidate.granularity, true);
        if (isoa_ws < candidate.metrics.working_set_bytes) {
            candidate.metrics.working_set_bytes = isoa_ws;
        }
    }

    if (std::getenv("MLSYS_OPTIMUS_DEBUG_TILE")) {
        std::cerr << "[cand] ops=[";
        for (size_t i = 0; i < candidate.ops.size(); ++i) {
            std::cerr << (i ? "," : "") << candidate.ops[i];
        }
        std::cerr << "] n_tiles=" << proxy_ranked.size()
                  << " valid=" << (candidate.metrics.valid ? 1 : 0)
                  << " gran=" << candidate.granularity.width << "x"
                  << candidate.granularity.height << "x"
                  << candidate.granularity.depth
                  << " lat=" << candidate.metrics.latency
                  << " reason=" << ToString(candidate.legality.reason) << "\n";
    }

    return candidate;
}

CandidateGroup BuildBestCandidate(const Problem& problem, const OpGraph& graph,
                                  size_t start, size_t end,
                                  const OptimusConfig& config) {
    std::vector<size_t> ops;
    for (size_t i = start; i <= end; ++i) {
        ops.push_back(graph.topo_order[i]);
    }
    return BuildBestCandidate(problem, graph, ops, config);
}

// ---------------------------------------------------------------------------
// Scheduler-extended candidate builder.  Delegates to BuildBestCandidate for
// the base logic, then optionally injects scheduler-proposed granularities and
// attaches a SchedulerEstimate to the winning candidate.
// ---------------------------------------------------------------------------
CandidateGroup BuildBestCandidateWithScheduler(
    const Problem& problem, const OpGraph& graph,
    const std::vector<size_t>& input_ops,
    const OptimusConfig& config) {

    CandidateGroup candidate;
    candidate.ops = input_ops;
    std::sort(candidate.ops.begin(), candidate.ops.end(),
              [&](size_t lhs, size_t rhs) {
                  return graph.topo_pos[lhs] < graph.topo_pos[rhs];
              });
    if (candidate.ops.empty()) {
        return candidate;
    }
    candidate.start = graph.topo_pos[candidate.ops.front()];
    candidate.end = graph.topo_pos[candidate.ops.back()];

    if (!SharesCommonOutputShape(problem, candidate.ops)) {
        return candidate;
    }
    if (!IsConnectedSubDAG(graph, candidate.ops)) {
        return candidate;
    }

    candidate.boundary = ComputeBoundary(problem, graph, candidate.ops);
    for (size_t tensor_id : candidate.boundary.internal_tensors) {
        candidate.internalized_bytes += TensorElements(problem.tensors[tensor_id]);
    }
    if (candidate.ops.size() > 1 && candidate.boundary.internal_tensors.empty()) {
        return candidate;
    }

    // --- Granularity enumeration (same as BuildBestCandidate) ----------------
    const auto& ref_out =
        problem.tensors[problem.ops[candidate.ops.front()].outputs.front()];
    const auto width_candidates =
        BuildSpatialCandidates(ref_out.width, problem.native_granularity.width);
    const auto height_candidates =
        BuildSpatialCandidates(ref_out.height, problem.native_granularity.height);
    const auto k_candidates = BuildKCandidates(problem, candidate.ops);
    std::vector<std::pair<double, Granularity>> proxy_ranked;
    const bool is_linear = IsLinearChainCandidate(graph, candidate.ops);

    for (int64_t width : width_candidates) {
        for (int64_t height : height_candidates) {
            for (int64_t depth : k_candidates) {
                Granularity granularity{width, height, depth};
                const int64_t working_set = ISOAAwareWorkingSetBytes(
                    problem, graph, candidate.ops, candidate.boundary,
                    granularity, is_linear);
                if (working_set > problem.fast_memory_capacity) {
                    continue;
                }
                double proxy_score = EstimateProxyLatency(
                    problem, graph, candidate.ops, candidate.boundary, granularity);
                if (config.guidance_mode == GuidanceMode::kConvAccelerator) {
                    if (config.conv_guidance_variant ==
                        ConvGuidanceVariant::kAdditivePenaltyV1) {
                        const ConvGuidanceMetrics conv_metrics =
                            AnalyzeConvGuidance(problem, graph, candidate.ops, granularity);
                        if (conv_metrics.valid) {
                            if (conv_metrics.working_set_bytes >
                                problem.fast_memory_capacity) {
                                continue;
                            }
                            proxy_score += 0.35 * conv_metrics.ranking_penalty;
                        }
                    }
                }
                proxy_ranked.push_back({proxy_score, granularity});
            }
        }
    }

    // --- NEW: Inject scheduler-proposed granularities ------------------------
    if (config.scheduler_propose_granularity) {
        auto scheduler_proposals = ProposeSchedulerGranularity(
            problem, graph, candidate.ops, candidate.boundary);
        for (const auto& proposed : scheduler_proposals) {
            // Check for duplicates.
            bool exists = false;
            for (const auto& [_, g] : proxy_ranked) {
                if (g.width == proposed.width && g.height == proposed.height &&
                    g.depth == proposed.depth) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                // Verify working set fits (ISOA-aware for linear chains).
                int64_t ws = ISOAAwareWorkingSetBytes(
                    problem, graph, candidate.ops, candidate.boundary,
                    proposed, is_linear);
                if (ws <= problem.fast_memory_capacity) {
                    double proxy_score = EstimateProxyLatency(
                        problem, graph, candidate.ops, candidate.boundary, proposed);
                    // Give scheduler proposals a slight priority boost (5%).
                    proxy_ranked.push_back({proxy_score * 0.95, proposed});
                }
            }
        }
    }

    // --- Analytical tile candidates (same as original) -----------------------
    {
        std::set<int64_t> tried_k;
        for (auto& [score, gran] : proxy_ranked) {
            tried_k.insert(gran.depth);
        }
        std::vector<int64_t> k_steps_to_try(tried_k.begin(), tried_k.end());
        if (k_steps_to_try.empty()) {
            k_steps_to_try = BuildKCandidates(problem, candidate.ops);
        }
        AddAnalyticalTileCandidates(problem, graph, candidate.ops,
                                     candidate.boundary, k_steps_to_try,
                                     &proxy_ranked);
    }

    // --- Sort and optionally apply V2 reranking ------------------------------
    std::sort(proxy_ranked.begin(), proxy_ranked.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    if (config.guidance_mode == GuidanceMode::kConvAccelerator &&
        config.conv_guidance_variant == ConvGuidanceVariant::kLocalRerankV2) {
        ApplyConvLocalRerankV2(problem, graph, candidate.ops, &proxy_ranked);
    }
    if (proxy_ranked.empty()) {
        return candidate;
    }

    // --- Evaluate top candidates with official scorer ------------------------
    const size_t budget = ComputeRefineBudget(proxy_ranked, candidate.ops.size());
    for (size_t i = 0; i < budget; ++i) {
        GroupMetrics metrics = EvaluateGroup(problem, graph, candidate.ops,
                                             candidate.boundary,
                                             proxy_ranked[i].second,
                                             nullptr);
        if (!metrics.valid) {
            continue;
        }
        if (!candidate.metrics.valid || metrics.latency < candidate.metrics.latency) {
            candidate.granularity = proxy_ranked[i].second;
            candidate.metrics = metrics;
        }
    }

    // --- NEW: Run scheduler analysis on the winning granularity ---------------
    if (candidate.metrics.valid) {
        candidate.scheduler = RunSchedulerAnalysis(
            problem, graph, candidate.ops, candidate.boundary,
            candidate.granularity);
    }

    // ISOA correction: if the evaluator-computed working set exceeds capacity,
    // try the ISOA-reduced estimate to allow the candidate to survive retain
    // choices in DP (IsCapacityFeasible uses candidate.metrics.working_set_bytes).
    if (candidate.metrics.valid && is_linear &&
        candidate.metrics.working_set_bytes > problem.fast_memory_capacity) {
        const int64_t isoa_ws = ISOAAwareWorkingSetBytes(
            problem, graph, candidate.ops, candidate.boundary,
            candidate.granularity, true);
        if (isoa_ws < candidate.metrics.working_set_bytes) {
            candidate.metrics.working_set_bytes = isoa_ws;
        }
    }

    return candidate;
}

CandidateGroup BuildBestCandidateWithScheduler(
    const Problem& problem, const OpGraph& graph,
    size_t start, size_t end, const OptimusConfig& config) {
    std::vector<size_t> ops;
    for (size_t i = start; i <= end; ++i) {
        ops.push_back(graph.topo_order[i]);
    }
    return BuildBestCandidateWithScheduler(problem, graph, ops, config);
}

size_t EstimateMaxGroupSize(const Problem& problem) {
    const int64_t native_tile =
        std::max<int64_t>(1, problem.native_granularity.width) *
        std::max<int64_t>(1, problem.native_granularity.height);
    if (native_tile <= 0) {
        return 4;
    }
    const int64_t ratio = problem.fast_memory_capacity / native_tile;
    if (ratio <= 2) {
        return 2;
    }
    if (ratio <= 4) {
        return 4;
    }
    if (ratio <= 8) {
        return 6;
    }
    return kDefaultMaxGroupSize;
}

std::vector<std::vector<CandidateGroup>> GenerateCandidates(
    const Problem& problem, const OpGraph& graph, const OptimusConfig& config) {
    constexpr size_t kReasonCount =
        static_cast<size_t>(LegalityReason::kRuntimeBudgetExceeded) + 1;
    const size_t n = graph.topo_order.size();
    const size_t max_group_size = EstimateMaxGroupSize(problem);
    std::vector<std::vector<CandidateGroup>> by_start(n);
    std::vector<size_t> rejected_candidates_by_start(n, 0);
    std::vector<std::vector<size_t>> reject_reasons_by_start(
        n, std::vector<size_t>(kReasonCount, 0));

    for (size_t start = 0; start < n; ++start) {
        for (size_t end = start;
             end < n && end < start + max_group_size; ++end) {
            CandidateGroup candidate =
                BuildBestCandidate(problem, graph, start, end, config);
            if (!candidate.metrics.valid) {
                ++rejected_candidates_by_start[start];
                ++reject_reasons_by_start[start][
                    static_cast<size_t>(candidate.legality.reason)];
                if (LegalityDebugEnabled()) {
                    std::cerr << "  legality_reject(interval) start=" << start
                              << " level=" << ToString(candidate.legality.failed_level)
                              << " reason=" << ToString(candidate.legality.reason)
                              << " note=" << candidate.legality.debug_note << "\n";
                }
                continue;
            }

            if (candidate.ops.size() > 1) {
                // Require at least 1 internalized element for multi-op
                // candidates.  The old threshold (native_w * native_h) was
                // far too conservative and rejected most fusion groups.
                constexpr int64_t min_gain = 1;
                if (candidate.internalized_bytes < min_gain) {
                    candidate.legality.is_valid = false;
                    candidate.legality.level_passed = LegalityLevel::kL1Execution;
                    candidate.legality.failed_level = LegalityLevel::kL3Policy;
                    candidate.legality.reason = LegalityReason::kHeuristicRejectPolicy;
                    candidate.legality.debug_note = "interval_min_internalized";
                    ++rejected_candidates_by_start[start];
                    ++reject_reasons_by_start[start][
                        static_cast<size_t>(candidate.legality.reason)];
                    if (LegalityDebugEnabled()) {
                        std::cerr << "  legality_reject(interval) start=" << start
                                  << " level=" << ToString(candidate.legality.failed_level)
                                  << " reason=" << ToString(candidate.legality.reason)
                                  << " note=" << candidate.legality.debug_note << "\n";
                    }
                    continue;
                }
            }

            by_start[start].push_back(std::move(candidate));
        }

        if (by_start[start].empty()) {
            CandidateGroup fallback =
                BuildBestCandidate(problem, graph, start, start, config);
            if (fallback.metrics.valid) {
                by_start[start].push_back(std::move(fallback));
            }
        }

        if (SeedDebugEnabled() || LegalityDebugEnabled()) {
            std::cerr << "interval start " << start
                      << " candidates=" << by_start[start].size()
                      << " rejected=" << rejected_candidates_by_start[start] << "\n";
            std::vector<std::pair<size_t, size_t>> reason_counts;
            for (size_t i = 0; i < reject_reasons_by_start[start].size(); ++i) {
                const size_t count = reject_reasons_by_start[start][i];
                if (count > 0) {
                    reason_counts.push_back({count, i});
                }
            }
            std::sort(reason_counts.begin(), reason_counts.end(),
                      [](const auto& lhs, const auto& rhs) {
                          return lhs.first > rhs.first;
                      });
            if (!reason_counts.empty()) {
                std::cerr << "  reject_reasons:";
                const size_t top_k = std::min<size_t>(3, reason_counts.size());
                for (size_t i = 0; i < top_k; ++i) {
                    const auto reason = static_cast<LegalityReason>(reason_counts[i].second);
                    std::cerr << " " << ToString(reason) << "=" << reason_counts[i].first;
                }
                std::cerr << "\n";
            }
        }
    }

    return by_start;
}

std::vector<size_t> CollectGrowthFrontier(
    const OpGraph& graph, const std::vector<size_t>& current_ops,
    bool allow_predecessor_growth, bool allow_successor_growth,
    size_t max_frontier) {
    std::unordered_set<size_t> current_set(current_ops.begin(), current_ops.end());
    std::vector<size_t> frontier;
    frontier.reserve(max_frontier);

    auto try_add = [&](size_t op_id) {
        if (current_set.count(op_id)) {
            return;
        }
        frontier.push_back(op_id);
    };

    for (size_t op_id : current_ops) {
        if (allow_successor_growth) {
            for (size_t succ : graph.succs[op_id]) {
                try_add(succ);
            }
        }
        if (allow_predecessor_growth) {
            for (size_t pred : graph.preds[op_id]) {
                try_add(pred);
            }
        }
    }

    frontier.erase(std::unique(frontier.begin(), frontier.end()), frontier.end());

    // Smart frontier sorting: prioritize nodes that connect to multiple branches
    // (higher degree = more growth opportunity) then by topo order
    std::sort(frontier.begin(), frontier.end(),
              [&](size_t lhs, size_t rhs) {
                  const size_t lhs_degree = graph.succs[lhs].size() + graph.preds[lhs].size();
                  const size_t rhs_degree = graph.succs[rhs].size() + graph.preds[rhs].size();
                  if (lhs_degree != rhs_degree) {
                      return lhs_degree > rhs_degree;  // Higher degree first
                  }
                  return graph.topo_pos[lhs] < graph.topo_pos[rhs];
              });

    if (frontier.size() > max_frontier) {
        frontier.resize(max_frontier);
    }
    return frontier;
}

bool IsContiguousTopoSpan(const OpGraph& graph, const CandidateGroup& candidate) {
    if (candidate.ops.empty()) {
        return false;
    }
    if (candidate.ops.size() != candidate.end - candidate.start + 1) {
        return false;
    }
    for (size_t i = 0; i < candidate.ops.size(); ++i) {
        if (candidate.ops[i] != graph.topo_order[candidate.start + i]) {
            return false;
        }
    }
    return true;
}

bool PassSeedPolicyFilter(const CandidateGroup& candidate,
                          const Problem& problem) {
    if (candidate.ops.size() <= 1) {
        return true;
    }

    const bool enforce_internalized_threshold = ReadBoolEnvOrDefault(
        "MLSYS_OPTIMUS_SEED_POLICY_ENFORCE_INTERNALIZED", true);
    if (!enforce_internalized_threshold) {
        return true;
    }

    const int64_t default_min_internalized =
        std::max<int64_t>(1, problem.native_granularity.width) *
        std::max<int64_t>(1, problem.native_granularity.height);
    const int64_t min_internalized = std::max<int64_t>(
        0, ReadInt64EnvOrDefault("MLSYS_OPTIMUS_SEED_POLICY_MIN_INTERNALIZED",
                                 default_min_internalized));
    return candidate.internalized_bytes >= min_internalized;
}

void RankAndTrimSeedCandidates(std::vector<CandidateGroup>* candidates,
                               size_t max_candidates) {
    if (candidates == nullptr) {
        return;
    }

    std::sort(candidates->begin(), candidates->end(),
              [](const CandidateGroup& lhs, const CandidateGroup& rhs) {
                  if (lhs.ops == rhs.ops) {
                      return lhs.metrics.latency < rhs.metrics.latency;
                  }
                  return lhs.ops < rhs.ops;
              });
    candidates->erase(
        std::unique(candidates->begin(), candidates->end(),
                    [](const CandidateGroup& lhs, const CandidateGroup& rhs) {
                        return lhs.ops == rhs.ops;
                    }),
        candidates->end());

    std::sort(candidates->begin(), candidates->end(),
              [](const CandidateGroup& lhs, const CandidateGroup& rhs) {
                  const double lhs_solo = static_cast<double>(lhs.ops.size());
                  const double rhs_solo = static_cast<double>(rhs.ops.size());
                  const double lhs_density = lhs.internalized_bytes -
                                             lhs.metrics.latency / lhs_solo;
                  const double rhs_density = rhs.internalized_bytes -
                                             rhs.metrics.latency / rhs_solo;
                  if (lhs_density != rhs_density) {
                      return lhs_density > rhs_density;
                  }
                  return lhs.metrics.latency < rhs.metrics.latency;
              });

    if (candidates->size() > max_candidates) {
        candidates->resize(max_candidates);
    }
}

std::vector<std::vector<CandidateGroup>> GenerateSeedGrowthCandidates(
    const Problem& problem, const OpGraph& graph, const OptimusConfig& config) {
    constexpr size_t kReasonCount =
        static_cast<size_t>(LegalityReason::kRuntimeBudgetExceeded) + 1;
    const size_t n = graph.topo_order.size();
    SeedGrowthRuntimeConfig runtime = GetSeedGrowthRuntimeConfig();
    // The bitmask/frontier DP handles non-contiguous groups correctly, so the
    // contiguous-span filter must be disabled when the caller opts in via
    // config.allow_noncontig_groups. Without this override, every
    // non-contiguous candidate is rejected at L3 (note=contiguous_required)
    // before the set-cover DP ever sees it.
    if (config.allow_noncontig_groups) {
        runtime.require_contiguous_topo = false;
    }
    const size_t max_states_per_seed =
        runtime.max_states_per_seed > 0
            ? runtime.max_states_per_seed
            : std::min<size_t>(1500, std::max<size_t>(400, n * 20));
    const size_t estimated_group_size = EstimateMaxGroupSize(problem);
    const size_t max_group_size = runtime.max_group_size_override > 0
                                      ? std::min(runtime.max_group_size_override,
                                                 std::max<size_t>(1, n))
                                      : estimated_group_size;
    std::vector<std::vector<CandidateGroup>> by_start(n);
    std::vector<size_t> explored_states_by_start(n, 0);
    std::vector<size_t> accepted_candidates_by_start(n, 0);
    std::vector<size_t> rejected_candidates_by_start(n, 0);
    std::vector<std::vector<size_t>> reject_reasons_by_start(
        n, std::vector<size_t>(kReasonCount, 0));
    std::unordered_map<std::vector<size_t>, CandidateGroup, OpVectorHash> candidate_cache;
    candidate_cache.reserve(std::max<size_t>(1024, runtime.total_queue_budget / 2));
    size_t global_queue_pushes = 0;

    // Pre-generate singleton candidates for ALL topo positions.
    // This guarantees full coverage: even if seed-growth doesn't produce
    // any candidate starting at a given position, the DP can always fall
    // back to the singleton.
    for (size_t pos = 0; pos < n; ++pos) {
        CandidateGroup singleton =
            BuildBestCandidate(problem, graph, {graph.topo_order[pos]}, config);
        if (singleton.metrics.valid) {
            by_start[pos].push_back(std::move(singleton));
        }
    }

    for (size_t topo_idx = 0; topo_idx < n; ++topo_idx) {
        const size_t seed = graph.topo_order[topo_idx];
        std::queue<std::vector<size_t>> pending;
        std::unordered_set<std::vector<size_t>, OpVectorHash> seen;
        size_t explored_states = 0;
        size_t accepted_candidates = 0;
        size_t rejected_candidates = 0;
        double best_latency = std::numeric_limits<double>::max();
        size_t iterations_without_improvement = 0;
        constexpr size_t kMaxIterationsWithoutImprovement = 20;  // More aggressive early termination

        const std::vector<size_t> seed_ops = CandidateKeyFromOps(
            CanonicalizeOpSet({seed}, graph));
        pending.push(seed_ops);
        ++global_queue_pushes;
        seen.insert(seed_ops);

        while (!pending.empty()) {
            std::vector<size_t> current_ops = pending.front();
            pending.pop();
            ++explored_states;

            if (explored_states > max_states_per_seed) {
                ++rejected_candidates;
                ++reject_reasons_by_start[topo_idx][static_cast<size_t>(
                    LegalityReason::kRuntimeBudgetExceeded)];
                if (LegalityDebugEnabled()) {
                    std::cerr << "  legality_reject start=" << topo_idx
                              << " level=" << ToString(LegalityLevel::kL3Policy)
                              << " reason=" << ToString(LegalityReason::kRuntimeBudgetExceeded)
                              << " note=seed_state_budget_exceeded\n";
                }
                break;
            }

            // Early termination: if no improvement for many iterations, skip this seed
            // (likely exhausted high-quality expansion directions)
            if (iterations_without_improvement > kMaxIterationsWithoutImprovement) {
                if (LegalityDebugEnabled()) {
                    std::cerr << "  seed_growth early_termination start=" << topo_idx
                              << " iterations_no_progress=" << iterations_without_improvement << "\n";
                }
                break;
            }

            CandidateGroup candidate;
            const auto cached = candidate_cache.find(current_ops);
            if (cached != candidate_cache.end()) {
                candidate = cached->second;
            } else {
                candidate = BuildBestCandidate(problem, graph, current_ops, config);
                candidate_cache.emplace(current_ops, candidate);
            }

            // Track progress: update iteration counter and best latency
            if (candidate.metrics.valid) {
                if (candidate.metrics.latency < best_latency) {
                    best_latency = candidate.metrics.latency;
                    iterations_without_improvement = 0;  // Reset counter
                } else {
                    ++iterations_without_improvement;
                }
            } else {
                ++iterations_without_improvement;
            }

            if (candidate.metrics.valid) {
                if (runtime.require_contiguous_topo &&
                    !IsContiguousTopoSpan(graph, candidate)) {
                    candidate.legality.is_valid = false;
                    candidate.legality.level_passed = LegalityLevel::kL1Execution;
                    candidate.legality.failed_level = LegalityLevel::kL3Policy;
                    candidate.legality.reason = LegalityReason::kHeuristicRejectPolicy;
                    candidate.legality.debug_note = "contiguous_required";
                    ++rejected_candidates;
                    ++reject_reasons_by_start[topo_idx][static_cast<size_t>(
                        candidate.legality.reason)];
                    if (LegalityDebugEnabled()) {
                        std::cerr << "  legality_reject start=" << topo_idx
                                  << " level=" << ToString(candidate.legality.failed_level)
                                  << " reason=" << ToString(candidate.legality.reason)
                                  << " note=" << candidate.legality.debug_note << "\n";
                    }
                    continue;
                }
                if (PassSeedPolicyFilter(candidate, problem)) {
                    // When allow_noncontig_groups is set, keep every accepted
                    // candidate (contiguous or not) and file it under the
                    // seed's topo-position so the set-cover DP can pick it up
                    // at the canonical smallest-topo-position. Otherwise only
                    // keep contiguous candidates because the position-indexed
                    // DP assumes next_start = candidate.end + 1 and would
                    // skip uncovered ops between start and end.
                    const bool is_contiguous =
                        (static_cast<size_t>(candidate.end - candidate.start + 1) ==
                         candidate.ops.size());
                    if (config.allow_noncontig_groups) {
                        // File under seed's topo-index; the DP rebuilds the
                        // canonical v0_topo_idx from ops, not from this key.
                        by_start[topo_idx].push_back(candidate);
                    } else if (is_contiguous) {
                        by_start[candidate.start].push_back(candidate);
                    }
                    ++accepted_candidates;
                } else {
                    candidate.legality.is_valid = false;
                    candidate.legality.level_passed = LegalityLevel::kL1Execution;
                    candidate.legality.failed_level = LegalityLevel::kL3Policy;
                    candidate.legality.reason = LegalityReason::kHeuristicRejectPolicy;
                    candidate.legality.debug_note = "seed_policy_filter";
                    ++rejected_candidates;
                    ++reject_reasons_by_start[topo_idx][static_cast<size_t>(
                        candidate.legality.reason)];
                    if (LegalityDebugEnabled()) {
                        std::cerr << "  legality_reject start=" << topo_idx
                                  << " level=" << ToString(candidate.legality.failed_level)
                                  << " reason=" << ToString(candidate.legality.reason)
                                  << " note=" << candidate.legality.debug_note << "\n";
                    }
                }
            } else {
                ++rejected_candidates;
                ++reject_reasons_by_start[topo_idx][static_cast<size_t>(
                    candidate.legality.reason)];
                if (LegalityDebugEnabled()) {
                    std::cerr << "  legality_reject start=" << topo_idx
                              << " level=" << ToString(candidate.legality.failed_level)
                              << " reason=" << ToString(candidate.legality.reason)
                              << " note=" << candidate.legality.debug_note << "\n";
                }
            }

            if (current_ops.size() >= max_group_size) {
                continue;
            }

            const std::vector<size_t> frontier = CollectGrowthFrontier(
                graph, current_ops, runtime.allow_predecessor_growth,
                runtime.allow_successor_growth, runtime.max_frontier);

            for (size_t next_op : frontier) {
                if (global_queue_pushes >= runtime.total_queue_budget) {
                    break;
                }
                std::vector<size_t> grown_ops = current_ops;
                grown_ops.push_back(next_op);
                const std::vector<size_t> canonical_ops = CandidateKeyFromOps(
                    CanonicalizeOpSet(grown_ops, graph));
                if (seen.insert(canonical_ops).second) {
                    pending.push(canonical_ops);
                    ++global_queue_pushes;
                }
            }
        }

        explored_states_by_start[topo_idx] = explored_states;
        accepted_candidates_by_start[topo_idx] = accepted_candidates;
        rejected_candidates_by_start[topo_idx] = rejected_candidates;
    }

    for (size_t start = 0; start < n; ++start) {
        auto& candidates = by_start[start];
        RankAndTrimSeedCandidates(&candidates, runtime.max_candidates_per_start);

        if (candidates.empty()) {
            CandidateGroup fallback =
                BuildBestCandidate(problem, graph, {graph.topo_order[start]}, config);
            if (fallback.metrics.valid) {
                candidates.push_back(std::move(fallback));
            }
        }

        if (SeedDebugEnabled()) {
            std::cerr << "seed start " << start << " candidates=" << candidates.size() << "\n";
            if (SeedDebugVerboseEnabled()) {
                for (const auto& candidate : candidates) {
                    std::cerr << "  ops:";
                    for (size_t op_id : candidate.ops) {
                        std::cerr << " " << op_id;
                    }
                    std::cerr << " lat=" << candidate.metrics.latency
                              << " internal=" << candidate.internalized_bytes
                              << " legality=" << ToString(candidate.legality.failed_level)
                              << "/" << ToString(candidate.legality.reason) << "\n";
                }
            }
            std::cerr << "  explored=" << explored_states_by_start[start]
                      << " accepted=" << accepted_candidates_by_start[start]
                      << " rejected=" << rejected_candidates_by_start[start] << "\n";

            std::vector<std::pair<size_t, size_t>> reason_counts;
            for (size_t i = 0; i < reject_reasons_by_start[start].size(); ++i) {
                const size_t count = reject_reasons_by_start[start][i];
                if (count > 0) {
                    reason_counts.push_back({count, i});
                }
            }
            std::sort(reason_counts.begin(), reason_counts.end(),
                      [](const auto& lhs, const auto& rhs) {
                          return lhs.first > rhs.first;
                      });
            if (!reason_counts.empty()) {
                std::cerr << "  reject_reasons:";
                const size_t top_k = std::min<size_t>(3, reason_counts.size());
                for (size_t i = 0; i < top_k; ++i) {
                    const auto reason = static_cast<LegalityReason>(reason_counts[i].second);
                    std::cerr << " " << ToString(reason) << "=" << reason_counts[i].first;
                }
                std::cerr << "\n";
            }
        }
    }

    return by_start;
}

std::vector<size_t> FilterRetainedInputsForCandidate(
    const CandidateGroup& candidate,
    const std::vector<size_t>& retained_inputs) {
    std::unordered_set<size_t> boundary_inputs(candidate.boundary.boundary_inputs.begin(),
                                               candidate.boundary.boundary_inputs.end());
    std::vector<size_t> filtered;
    for (size_t tensor_id : retained_inputs) {
        if (boundary_inputs.count(tensor_id)) {
            filtered.push_back(tensor_id);
        }
    }
    std::sort(filtered.begin(), filtered.end());
    filtered.erase(std::unique(filtered.begin(), filtered.end()), filtered.end());
    return filtered;
}

std::vector<size_t> CanonicalizeRetainedInputs(
    const std::vector<size_t>& retained_inputs,
    const std::unordered_set<size_t>& useful_inputs) {
    std::vector<size_t> canonical;
    canonical.reserve(retained_inputs.size());
    for (size_t tensor_id : retained_inputs) {
        if (useful_inputs.count(tensor_id)) {
            canonical.push_back(tensor_id);
        }
    }
    std::sort(canonical.begin(), canonical.end());
    canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());
    return canonical;
}

std::vector<std::vector<size_t>> EnumerateRetainChoices(
    const Problem& problem, const CandidateGroup& current,
    const CandidateGroup& next, const std::vector<size_t>& retained_inputs) {
    const int64_t retained_in_bytes =
        TensorBytesForSet(problem, retained_inputs);
    const int64_t current_limit = problem.fast_memory_capacity -
                                  current.metrics.working_set_bytes -
                                  retained_in_bytes;
    const int64_t next_limit =
        problem.fast_memory_capacity - next.metrics.working_set_bytes;
    const int64_t available_bytes = std::min(current_limit, next_limit);

    std::unordered_set<size_t> next_inputs(next.boundary.boundary_inputs.begin(),
                                           next.boundary.boundary_inputs.end());
    std::vector<size_t> shareable_outputs;
    for (size_t tensor_id : current.boundary.boundary_outputs) {
        if (next_inputs.count(tensor_id)) {
            shareable_outputs.push_back(tensor_id);
        }
    }
    std::sort(shareable_outputs.begin(), shareable_outputs.end());

    std::vector<std::vector<size_t>> choices;
    choices.push_back({});
    if (shareable_outputs.empty() || available_bytes <= 0) {
        return choices;
    }

    if (shareable_outputs.size() <= 12) {
        const size_t total = static_cast<size_t>(1) << shareable_outputs.size();
        for (size_t mask = 1; mask < total; ++mask) {
            int64_t bytes = 0;
            std::vector<size_t> chosen;
            for (size_t i = 0; i < shareable_outputs.size(); ++i) {
                if ((mask >> i) & 1U) {
                    bytes += TensorElements(problem.tensors[shareable_outputs[i]]);
                    chosen.push_back(shareable_outputs[i]);
                }
            }
            if (bytes <= available_bytes) {
                choices.push_back(std::move(chosen));
            }
        }
    } else {
        std::sort(shareable_outputs.begin(), shareable_outputs.end(),
                  [&](size_t lhs, size_t rhs) {
                      return TensorElements(problem.tensors[lhs]) >
                             TensorElements(problem.tensors[rhs]);
                  });
        std::vector<size_t> prefix_choice;
        int64_t prefix_bytes = 0;
        for (size_t tensor_id : shareable_outputs) {
            const int64_t tensor_bytes = TensorElements(problem.tensors[tensor_id]);
            if (prefix_bytes + tensor_bytes > available_bytes) {
                continue;
            }
            prefix_bytes += tensor_bytes;
            prefix_choice.push_back(tensor_id);
            std::vector<size_t> chosen = prefix_choice;
            std::sort(chosen.begin(), chosen.end());
            choices.push_back(std::move(chosen));
        }
        for (size_t tensor_id : shareable_outputs) {
            const int64_t tensor_bytes = TensorElements(problem.tensors[tensor_id]);
            if (tensor_bytes <= available_bytes) {
                choices.push_back({tensor_id});
            }
        }
    }

    std::sort(choices.begin(), choices.end());
    choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
    return choices;
}

bool IsCapacityFeasible(const Problem& problem, const CandidateGroup& current,
                        const std::vector<size_t>& retained_inputs,
                        const std::vector<size_t>& retained_outputs) {
    const int64_t retained_in_bytes =
        TensorBytesForSet(problem, retained_inputs);
    const int64_t retained_out_bytes =
        TensorBytesForSet(problem, retained_outputs);
    return current.metrics.working_set_bytes + retained_in_bytes +
               retained_out_bytes <=
           problem.fast_memory_capacity;
}

SearchDecision SolveFromState(
    const Problem& problem, const OpGraph& graph,
    const std::vector<std::vector<CandidateGroup>>& candidates,
    const std::vector<std::unordered_set<size_t>>& useful_inputs_by_start,
    size_t start, const std::vector<size_t>& retained_inputs,
    std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash>* memo) {
    const size_t n = candidates.size();
    if (start >= n) {
        SearchDecision terminal;
        terminal.valid = retained_inputs.empty();
        terminal.cost = terminal.valid ? 0.0 : kInfinity;
        return terminal;
    }

    const SearchStateKey key{
        start, CanonicalizeRetainedInputs(retained_inputs, useful_inputs_by_start[start])};
    auto it = memo->find(key);
    if (it != memo->end()) {
        return it->second;
    }

    SearchDecision best;

    for (size_t cand_idx = 0; cand_idx < candidates[start].size(); ++cand_idx) {
        const auto& candidate = candidates[start][cand_idx];
        const std::vector<size_t> incoming_used =
            FilterRetainedInputsForCandidate(candidate, key.retained_inputs);
        const size_t next_start = candidate.end + 1;

        if (next_start >= n) {
            if (!IsCapacityFeasible(problem, candidate, incoming_used, {})) {
                continue;
            }
                const double current_cost =
                [&]() {
                    GroupMetrics metrics;
                    if (!EvaluateWithOfficialScorer(problem, candidate.ops,
                                                    candidate.granularity,
                                                    incoming_used, {}, &metrics)) {
                        return kInfinity;
                    }
                    return metrics.latency;
                }();
            if (current_cost < best.cost) {
                best.valid = true;
                best.cost = current_cost;
                best.candidate_index = cand_idx;
                best.retained_outputs.clear();
            }
            continue;
        }

        for (const auto& next_candidate : candidates[next_start]) {
            const auto retain_choices = EnumerateRetainChoices(
                problem, candidate, next_candidate, incoming_used);
            for (const auto& retained_outputs : retain_choices) {
                if (!IsCapacityFeasible(problem, candidate, incoming_used,
                                        retained_outputs)) {
                    continue;
                }
                GroupMetrics scored_metrics;
                if (!EvaluateWithOfficialScorer(problem, candidate.ops,
                                                candidate.granularity,
                                                incoming_used, retained_outputs,
                                                &scored_metrics)) {
                    continue;
                }
                const double current_cost = scored_metrics.latency;
                const SearchDecision future = SolveFromState(
                    problem, graph, candidates, useful_inputs_by_start, next_start,
                    retained_outputs, memo);
                if (!future.valid) {
                    continue;
                }

                const double total_cost = current_cost + future.cost;
                if (total_cost < best.cost) {
                    best.valid = true;
                    best.cost = total_cost;
                    best.candidate_index = cand_idx;
                    best.retained_outputs = retained_outputs;
                }
            }
        }
    }

    (*memo)[key] = best;
    return best;
}

// ---------------------------------------------------------------------------
// Scheduler-aware DP: same as SolveFromState but uses blended cost.
// ---------------------------------------------------------------------------
SearchDecision SolveFromStateWithScheduler(
    const Problem& problem, const OpGraph& graph,
    const std::vector<std::vector<CandidateGroup>>& candidates,
    const std::vector<std::unordered_set<size_t>>& useful_inputs_by_start,
    size_t start, const std::vector<size_t>& retained_inputs,
    const OptimusConfig& config,
    std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash>* memo) {
    const size_t n = candidates.size();
    if (start >= n) {
        SearchDecision terminal;
        terminal.valid = retained_inputs.empty();
        terminal.cost = terminal.valid ? 0.0 : kInfinity;
        return terminal;
    }

    const SearchStateKey key{
        start, CanonicalizeRetainedInputs(retained_inputs, useful_inputs_by_start[start])};
    auto it = memo->find(key);
    if (it != memo->end()) {
        return it->second;
    }

    SearchDecision best;

    for (size_t cand_idx = 0; cand_idx < candidates[start].size(); ++cand_idx) {
        const auto& candidate = candidates[start][cand_idx];
        const std::vector<size_t> incoming_used =
            FilterRetainedInputsForCandidate(candidate, key.retained_inputs);
        const size_t next_start = candidate.end + 1;

        if (next_start >= n) {
            if (!IsCapacityFeasible(problem, candidate, incoming_used, {})) {
                continue;
            }
            const double current_cost =
                [&]() {
                    GroupMetrics metrics;
                    if (!EvaluateWithOfficialScorer(problem, candidate.ops,
                                                    candidate.granularity,
                                                    incoming_used, {}, &metrics)) {
                        return kInfinity;
                    }
                    return ComputeBlendedCost(metrics.latency,
                                              candidate.scheduler,
                                              config.scheduler_cost_weight);
                }();
            if (current_cost < best.cost) {
                best.valid = true;
                best.cost = current_cost;
                best.candidate_index = cand_idx;
                best.retained_outputs.clear();
            }
            continue;
        }

        for (const auto& next_candidate : candidates[next_start]) {
            const auto retain_choices = EnumerateRetainChoices(
                problem, candidate, next_candidate, incoming_used);
            for (const auto& retained_outputs : retain_choices) {
                if (!IsCapacityFeasible(problem, candidate, incoming_used,
                                        retained_outputs)) {
                    continue;
                }
                GroupMetrics scored_metrics;
                if (!EvaluateWithOfficialScorer(problem, candidate.ops,
                                                candidate.granularity,
                                                incoming_used, retained_outputs,
                                                &scored_metrics)) {
                    continue;
                }
                const double current_cost = ComputeBlendedCost(
                    scored_metrics.latency, candidate.scheduler,
                    config.scheduler_cost_weight);
                const SearchDecision future = SolveFromStateWithScheduler(
                    problem, graph, candidates, useful_inputs_by_start, next_start,
                    retained_outputs, config, memo);
                if (!future.valid) {
                    continue;
                }

                const double total_cost = current_cost + future.cost;
                if (total_cost < best.cost) {
                    best.valid = true;
                    best.cost = total_cost;
                    best.candidate_index = cand_idx;
                    best.retained_outputs = retained_outputs;
                }
            }
        }
    }

    (*memo)[key] = best;
    return best;
}

Solution BuildSolutionFromSchedule(
    const Problem& problem, const std::vector<CandidateGroup>& schedule) {
    Solution solution;
    std::vector<size_t> retained_inputs;

    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto& candidate = schedule[i];
        const std::vector<size_t> incoming_used =
            FilterRetainedInputsForCandidate(candidate, retained_inputs);
        Subgraph subgraph;
        subgraph.op_ids = candidate.ops;
        subgraph.granularity = candidate.granularity;
        if (i + 1 < schedule.size()) {
            const auto retain_choices = EnumerateRetainChoices(
                problem, candidate, schedule[i + 1], incoming_used);
            if (!retain_choices.empty()) {
                subgraph.tensors_to_retain = retain_choices.back();
            }
        }
        subgraph.traversal_order = std::nullopt;
        GroupMetrics scored_metrics;
        if (!EvaluateWithOfficialScorer(problem, candidate.ops,
                                        candidate.granularity,
                                        incoming_used,
                                        subgraph.tensors_to_retain,
                                        &scored_metrics)) {
            break;
        }
        subgraph.subgraph_latency = scored_metrics.latency;
        solution.subgraphs.push_back(std::move(subgraph));

        retained_inputs = solution.subgraphs.back().tensors_to_retain;
    }

    return solution;
}

Solution BuildSolutionFromSearch(
    const Problem& problem, const OpGraph& /*graph*/,
    const std::vector<std::vector<CandidateGroup>>& candidates,
    const std::vector<std::unordered_set<size_t>>& useful_inputs_by_start,
    const std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash>& memo) {
    std::vector<CandidateGroup> schedule;
    size_t start = 0;
    std::vector<size_t> retained_inputs;

    while (start < candidates.size()) {
        const SearchStateKey key{
            start,
            CanonicalizeRetainedInputs(retained_inputs, useful_inputs_by_start[start])};
        auto it = memo.find(key);
        if (it == memo.end() || !it->second.valid ||
            it->second.candidate_index >= candidates[start].size()) {
            break;
        }
        schedule.push_back(candidates[start][it->second.candidate_index]);
        retained_inputs = it->second.retained_outputs;
        start = schedule.back().end + 1;
    }

    return BuildSolutionFromSchedule(problem, schedule);
}

// ============================================================================
// PAPER ALGORITHM IMPLEMENTATIONS: Optimus (Cai et al., TECS 2022)
// ============================================================================

// ---- Point 3: ISValid from Algorithm 1 ----

// Forward BFS from start_node following successor edges.
// Returns true if any node in target_set is reachable.
bool CanReachOpSet(const OpGraph& graph,
                   size_t start_node,
                   const std::unordered_set<size_t>& target_set) {
    std::queue<size_t> q;
    std::unordered_set<size_t> visited;
    q.push(start_node);
    visited.insert(start_node);
    while (!q.empty()) {
        size_t node = q.front();
        q.pop();
        if (target_set.count(node)) {
            return true;
        }
        for (size_t succ : graph.succs[node]) {
            if (!visited.count(succ)) {
                visited.insert(succ);
                q.push(succ);
            }
        }
    }
    return false;
}

// Paper Algorithm 1, ISValid (lines 24-30):
// Returns true if it is valid to fuse v0 into the existing group G'.
// Invalid when any successor of v0 is outside G' but can reach G' through
// an external path, which would create a cycle in the partition DAG.
bool IsFusionValid(const OpGraph& graph,
                   const std::unordered_set<size_t>& group_set,
                   size_t v0) {
    for (size_t c : graph.succs[v0]) {
        if (group_set.count(c)) {
            continue;  // c is in the group, no issue
        }
        if (CanReachOpSet(graph, c, group_set)) {
            return false;
        }
    }
    return true;
}

// ---- Points 4 & 6: ISOA + Analytical tile sizing (Paper Section 5.2, Eq. 4) ----
// With ISOA, only one spatial tile of each intermediate tensor is live at a time.
// Buffer formula: area_coeff * tile_h * tile_w
//                + row_coeff * tile_h * k_step   (MatMul LHS slices)
//                + col_coeff * tile_w * k_step   (MatMul RHS / weight slices)
// This models the parameter-refill cost (Point 5) via col_coeff * k_step.
// (TileBufferCoeffs struct is declared earlier for forward-reference use.)

TileBufferCoeffs ComputeBufferCoeffs(const Problem& problem,
                                      const OpGraph& graph,
                                      const std::vector<size_t>& ops,
                                      const GroupBoundary& boundary) {
    TileBufferCoeffs coeffs;
    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    std::unordered_set<size_t> boundary_out_set(boundary.boundary_outputs.begin(),
                                                 boundary.boundary_outputs.end());
    for (size_t op_id : ops) {
        const auto& op = problem.ops[op_id];
        if (IsMatMul(op)) {
            // LHS (activation slice): tile_h * k_step per step
            if (!op.inputs.empty()) {
                int p = graph.producer_of_tensor[op.inputs[0]];
                if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                    coeffs.row_coeff += 1.0;
                }
            }
            // RHS (weight slice, refill cost — Point 5): tile_w * k_step per step
            if (op.inputs.size() >= 2) {
                int p = graph.producer_of_tensor[op.inputs[1]];
                if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                    coeffs.col_coeff += 1.0;
                }
            }
            // Output partial sum accumulator: tile_h * tile_w
            coeffs.area_coeff += 1.0;
        } else if (IsPointwise(op)) {
            // Each external boundary input: tile_h * tile_w
            // (with ISOA, only the current tile is live — not the full tensor)
            for (size_t t : op.inputs) {
                int p = graph.producer_of_tensor[t];
                if (p < 0 || !op_set.count(static_cast<size_t>(p))) {
                    coeffs.area_coeff += 1.0;
                }
            }
            // Output tile: tile_h * tile_w
            if (boundary_out_set.count(op.outputs.front())) {
                coeffs.area_coeff += 1.0;
            }
        }
    }
    return coeffs;
}

// Solve: A_r * x^2 + B * x - C <= 0, find largest integer x >= 1.
// A_r = area_coeff * ratio,  B = (row*ratio + col) * k_step,  C = fast_memory.
// Returns 0 if infeasible.
int64_t SolveQuadraticTileWidth(double A_r, double B, double C) {
    if (C <= 0.0) {
        return 0;
    }
    if (A_r <= 0.0) {
        // Linear case: B * x <= C
        if (B <= 0.0) {
            return static_cast<int64_t>(1e15);
        }
        return static_cast<int64_t>(C / B);
    }
    // Quadratic: A_r * x^2 + B * x - C <= 0
    // x = (-B + sqrt(B^2 + 4*A_r*C)) / (2*A_r)
    double disc = B * B + 4.0 * A_r * C;
    if (disc < 0.0) {
        return 0;
    }
    double x = (-B + std::sqrt(disc)) / (2.0 * A_r);
    if (x < 1.0) {
        return 0;
    }
    return static_cast<int64_t>(x);
}

// Add analytically-derived max-tile candidates (Paper Eq. 4) to proxy_ranked.
// For each k_step and several aspect ratios, solves the quadratic for max tile_w,
// then adds the resulting Granularity if it passes the working-set check.
void AddAnalyticalTileCandidates(const Problem& problem,
                                  const OpGraph& graph,
                                  const std::vector<size_t>& ops,
                                  const GroupBoundary& boundary,
                                  const std::vector<int64_t>& k_steps,
                                  std::vector<std::pair<double, Granularity>>* proxy_ranked) {
    if (ops.empty() || proxy_ranked == nullptr) {
        return;
    }
    const auto& ref_out = problem.tensors[problem.ops[ops.front()].outputs.front()];
    const int64_t out_h = std::max<int64_t>(1, ref_out.height);
    const int64_t out_w = std::max<int64_t>(1, ref_out.width);
    const double C = static_cast<double>(problem.fast_memory_capacity);

    const TileBufferCoeffs coeffs = ComputeBufferCoeffs(problem, graph, ops, boundary);

    // Build aspect ratios: natural, 1:1, inverse, plus intermediates.
    const double natural = static_cast<double>(out_h) / static_cast<double>(out_w);
    const double inverse = static_cast<double>(out_w) / static_cast<double>(out_h);
    std::vector<double> ratios = {natural};
    if (out_h != out_w) {
        ratios.push_back(1.0);
    }
    ratios.push_back(inverse);
    for (double r : {0.25, 0.5, 2.0, 4.0}) {
        ratios.push_back(r);
    }
    if (natural > 0.0) ratios.push_back(std::sqrt(natural));
    if (inverse > 0.0) ratios.push_back(std::sqrt(inverse));
    // Deduplicate ratios.
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end(),
        [](double a, double b) { return std::abs(a - b) < 1e-9; }),
        ratios.end());

    for (int64_t k_step : k_steps) {
        if (k_step <= 0) {
            continue;
        }
        for (double r : ratios) {
            if (r <= 0.0) {
                continue;
            }
            const double A_r = coeffs.area_coeff * r;
            const double B   = (coeffs.row_coeff * r + coeffs.col_coeff) *
                               static_cast<double>(k_step);
            int64_t max_tw = SolveQuadraticTileWidth(A_r, B, C);
            if (max_tw <= 0) {
                continue;
            }
            int64_t tile_w = std::min<int64_t>(out_w, max_tw);
            int64_t tile_h = std::min<int64_t>(
                out_h, static_cast<int64_t>(r * static_cast<double>(tile_w)));
            tile_h = std::max<int64_t>(1, tile_h);
            tile_w = std::max<int64_t>(1, tile_w);

            Granularity g{tile_w, tile_h, k_step};
            // Binary search for largest fitting tile_w when initial guess overflows.
            int64_t ws = EstimateWorkingSetBytes(problem, graph, ops, boundary, g);
            if (ws > problem.fast_memory_capacity) {
                int64_t lo = 1, hi = tile_w;
                while (lo < hi) {
                    const int64_t mid = lo + (hi - lo + 1) / 2;
                    const int64_t th = std::max<int64_t>(1,
                        std::min<int64_t>(out_h,
                            static_cast<int64_t>(r * static_cast<double>(mid))));
                    const Granularity test_g{mid, th, k_step};
                    if (EstimateWorkingSetBytes(problem, graph, ops, boundary,
                                                test_g) <=
                        problem.fast_memory_capacity) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                tile_w = lo;
                tile_h = std::max<int64_t>(1,
                    std::min<int64_t>(out_h,
                        static_cast<int64_t>(r * static_cast<double>(tile_w))));
                g = {tile_w, tile_h, k_step};
                ws = EstimateWorkingSetBytes(problem, graph, ops, boundary, g);
                if (ws > problem.fast_memory_capacity) {
                    continue;
                }
            }
            const double proxy_score =
                EstimateProxyLatency(problem, graph, ops, boundary, g);
            if (std::isfinite(proxy_score)) {
                proxy_ranked->push_back({proxy_score, g});
            }
        }
    }
}

// ---- Point 3: Bitmask DP for small N ----

// A precomputed group candidate for the bitmask DP.
// mask is indexed by topo_order position (bit i = op at topo_order[i]).
struct BitmaskGroup {
    uint64_t mask     = 0;
    int v0_topo_idx   = -1;  // smallest topo position in the group
    CandidateGroup candidate;
};

// Convert existing position-indexed candidates to BitmaskGroups.
// Also checks IsFusionValid to filter out groups that create partition-DAG cycles.
std::vector<BitmaskGroup> BuildBitmaskGroups(
    const OpGraph& graph,
    const std::vector<std::vector<CandidateGroup>>& candidates) {
    const size_t n = graph.topo_order.size();
    std::vector<BitmaskGroup> result;
    std::unordered_set<uint64_t> seen_masks;
    size_t partition_cycle_rejects = 0;

    for (size_t start = 0; start < n; ++start) {
        for (const auto& cand : candidates[start]) {
            if (!cand.metrics.valid) {
                continue;
            }
            // Compute bitmask over topo positions and derive the canonical
            // v0_topo_idx (smallest topo position in the group) from ops,
            // NOT from `start`. Seed-growth may file a non-contiguous
            // candidate under the seed's topo-index, which is not
            // necessarily the smallest topo position in the group.
            uint64_t mask = 0;
            int v0_topo_idx = std::numeric_limits<int>::max();
            for (size_t op_id : cand.ops) {
                const int pos = static_cast<int>(graph.topo_pos[op_id]);
                mask |= (1ULL << pos);
                if (pos < v0_topo_idx) {
                    v0_topo_idx = pos;
                }
            }
            if (!seen_masks.insert(mask).second) {
                continue;
            }

            // ISValid check (Point 3): reject groups with partition-DAG cycles.
            // For each op in the group acting as v0, verify adding it doesn't
            // create a cyclic dependency.
            std::unordered_set<size_t> group_set(cand.ops.begin(), cand.ops.end());
            bool is_valid_partition = true;
            for (size_t op_id : cand.ops) {
                // Treat each op as if it were the "new" op being fused in
                std::unordered_set<size_t> rest_of_group = group_set;
                rest_of_group.erase(op_id);
                if (rest_of_group.empty()) {
                    break;  // single-op group always valid
                }
                if (!IsFusionValid(graph, rest_of_group, op_id)) {
                    is_valid_partition = false;
                    break;
                }
            }
            if (!is_valid_partition) {
                ++partition_cycle_rejects;
                if (LegalityDebugEnabled()) {
                    std::cerr << "  legality_reject(bitmask) start=" << start
                              << " level=" << ToString(LegalityLevel::kL0Graph)
                              << " reason=" << ToString(LegalityReason::kPartitionCycle)
                              << " note=is_fusion_valid_failed\n";
                }
                continue;
            }

            BitmaskGroup bg;
            bg.mask = mask;
            bg.v0_topo_idx = v0_topo_idx;
            bg.candidate = cand;
            result.push_back(std::move(bg));
        }
    }
    if (LegalityDebugEnabled() && partition_cycle_rejects > 0) {
        std::cerr << "bitmask rejects PARTITION_CYCLE=" << partition_cycle_rejects << "\n";
    }
    return result;
}

struct BitmaskDecision {
    bool   valid     = false;
    double cost      = kInfinity;
    size_t group_idx = 0;
};

// Recursive bitmask DP with memoization.
// covered: bitmask of topo positions already scheduled.
// full_mask: bitmask with all n bits set.
// groups_by_v0[i]: indices into groups for groups whose v0_topo_idx == i.
double SolveBitmaskDPImpl(
    const Problem& problem,
    const OpGraph& graph,
    const std::vector<BitmaskGroup>& groups,
    const std::vector<std::vector<size_t>>& groups_by_v0,
    size_t n,
    uint64_t covered,
    uint64_t full_mask,
    std::unordered_map<uint64_t, BitmaskDecision>* memo) {
    if (covered == full_mask) {
        return 0.0;
    }
    auto it = memo->find(covered);
    if (it != memo->end()) {
        return it->second.valid ? it->second.cost : kInfinity;
    }

    // Find the first uncovered op in topo order whose all predecessors are covered.
    int v0_topo_idx = -1;
    for (size_t i = 0; i < n; ++i) {
        if ((covered >> i) & 1u) {
            continue;
        }
        size_t op_id = graph.topo_order[i];
        bool ready = true;
        for (size_t pred : graph.preds[op_id]) {
            if (!((covered >> graph.topo_pos[pred]) & 1u)) {
                ready = false;
                break;
            }
        }
        if (ready) {
            v0_topo_idx = static_cast<int>(i);
            break;
        }
    }
    if (v0_topo_idx < 0) {
        (*memo)[covered] = {};
        return kInfinity;
    }

    BitmaskDecision best;
    if (static_cast<size_t>(v0_topo_idx) < groups_by_v0.size()) {
        for (size_t group_idx : groups_by_v0[v0_topo_idx]) {
            const BitmaskGroup& bg = groups[group_idx];
            // Group must not overlap already-covered ops
            if (bg.mask & covered) {
                continue;
            }
            // All ops in group must be ready (their preds are covered or internal)
            bool group_ready = true;
            for (size_t i = 0; i < n; ++i) {
                if (!((bg.mask >> i) & 1u)) {
                    continue;
                }
                size_t op_id = graph.topo_order[i];
                for (size_t pred : graph.preds[op_id]) {
                    size_t pred_topo = graph.topo_pos[pred];
                    if (!((covered >> pred_topo) & 1u) &&
                        !((bg.mask >> pred_topo) & 1u)) {
                        group_ready = false;
                        break;
                    }
                }
                if (!group_ready) {
                    break;
                }
            }
            if (!group_ready) {
                continue;
            }

            const double group_cost = bg.candidate.metrics.latency;
            const uint64_t new_covered = covered | bg.mask;
            const double future = SolveBitmaskDPImpl(
                problem, graph, groups, groups_by_v0, n,
                new_covered, full_mask, memo);
            const double total = group_cost + future;
            if (total < best.cost) {
                best.valid     = true;
                best.cost      = total;
                best.group_idx = group_idx;
            }
        }
    }

    (*memo)[covered] = best;
    return best.valid ? best.cost : kInfinity;
}

// Recover the ordered schedule from the bitmask DP memo.
std::vector<CandidateGroup> RecoverBitmaskSchedule(
    const std::vector<BitmaskGroup>& groups,
    uint64_t full_mask,
    const std::unordered_map<uint64_t, BitmaskDecision>& memo) {
    std::vector<CandidateGroup> schedule;
    uint64_t covered = 0;
    while (covered != full_mask) {
        auto it = memo.find(covered);
        if (it == memo.end() || !it->second.valid) {
            break;
        }
        const BitmaskGroup& bg = groups[it->second.group_idx];
        schedule.push_back(bg.candidate);
        covered |= bg.mask;
    }
    return schedule;
}

// ---- Frontier DP for N > 20 (set-cover DP with arbitrary-width bitset) ----

// Wide bitmask stored as a small vector of 64-bit words, indexed by topo
// position (bit i = op at topo_order[i]).  Mirrors BitmaskGroup but without
// the 64-bit ceiling.
struct FrontierGroup {
    std::vector<uint64_t> mask;
    int v0_topo_idx = -1;  // smallest topo position in the group
    CandidateGroup candidate;
};

struct FrontierMaskHash {
    size_t operator()(const std::vector<uint64_t>& m) const noexcept {
        // FNV-1a mix on the 64-bit words; good enough for memo keys.
        size_t h = 1469598103934665603ULL;
        for (uint64_t w : m) {
            h ^= static_cast<size_t>(w);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct FrontierDecision {
    bool   valid     = false;
    double cost      = kInfinity;
    size_t group_idx = 0;
};

// Returns the number of 64-bit words required to store `n` topo positions.
static inline size_t FrontierNumWords(size_t n) { return (n + 63) / 64; }

static inline bool FrontierTestBit(const std::vector<uint64_t>& m, size_t i) {
    return (m[i >> 6] >> (i & 63)) & 1u;
}

static inline void FrontierSetBit(std::vector<uint64_t>& m, size_t i) {
    m[i >> 6] |= (1ULL << (i & 63));
}

static inline bool FrontierAnyOverlap(const std::vector<uint64_t>& a,
                                      const std::vector<uint64_t>& b) {
    const size_t w = std::min(a.size(), b.size());
    for (size_t i = 0; i < w; ++i) {
        if (a[i] & b[i]) return true;
    }
    return false;
}

static inline void FrontierOrInto(std::vector<uint64_t>& dst,
                                  const std::vector<uint64_t>& src) {
    for (size_t i = 0; i < dst.size() && i < src.size(); ++i) {
        dst[i] |= src[i];
    }
}

// Convert position-indexed candidates to FrontierGroups, analogous to
// BuildBitmaskGroups but without the 64-op ceiling.
static std::vector<FrontierGroup> BuildFrontierGroups(
    const OpGraph& graph,
    const std::vector<std::vector<CandidateGroup>>& candidates) {
    const size_t n = graph.topo_order.size();
    const size_t W = FrontierNumWords(n);
    std::vector<FrontierGroup> result;
    std::unordered_set<std::vector<uint64_t>, FrontierMaskHash> seen_masks;
    size_t partition_cycle_rejects = 0;

    for (size_t start = 0; start < n; ++start) {
        for (const auto& cand : candidates[start]) {
            if (!cand.metrics.valid) continue;

            std::vector<uint64_t> mask(W, 0ULL);
            int v0_topo_idx = std::numeric_limits<int>::max();
            for (size_t op_id : cand.ops) {
                const int pos = static_cast<int>(graph.topo_pos[op_id]);
                FrontierSetBit(mask, static_cast<size_t>(pos));
                if (pos < v0_topo_idx) v0_topo_idx = pos;
            }
            if (!seen_masks.insert(mask).second) continue;

            // Reject partition-DAG cycles (mirrors BuildBitmaskGroups).
            std::unordered_set<size_t> group_set(cand.ops.begin(), cand.ops.end());
            bool is_valid_partition = true;
            for (size_t op_id : cand.ops) {
                std::unordered_set<size_t> rest = group_set;
                rest.erase(op_id);
                if (rest.empty()) break;
                if (!IsFusionValid(graph, rest, op_id)) {
                    is_valid_partition = false;
                    break;
                }
            }
            if (!is_valid_partition) {
                ++partition_cycle_rejects;
                continue;
            }

            FrontierGroup fg;
            fg.mask = std::move(mask);
            fg.v0_topo_idx = v0_topo_idx;
            fg.candidate = cand;
            result.push_back(std::move(fg));
        }
    }
    if (LegalityDebugEnabled() && partition_cycle_rejects > 0) {
        std::cerr << "frontier rejects PARTITION_CYCLE="
                  << partition_cycle_rejects << "\n";
    }
    return result;
}

// Set-cover DP for N > 20.  State = `covered` bitset over topo positions.
// At each state we pick the first uncovered op whose predecessors are all
// covered (deterministic frontier choice) and enumerate groups whose
// smallest-topo-position op equals that pick.  This mirrors SolveBitmaskDPImpl
// but uses a vector<uint64_t> state so N is not capped at 64.
//
// Safety: on memoization-cap overflow, returns kInfinity so the caller can
// fall back to the positional DP.
static double SolveFrontierDPImpl(
    const Problem& problem,
    const OpGraph& graph,
    const std::vector<FrontierGroup>& groups,
    const std::vector<std::vector<size_t>>& groups_by_v0,
    size_t n,
    const std::vector<uint64_t>& covered,
    const std::vector<uint64_t>& full_mask,
    size_t memo_cap,
    std::unordered_map<std::vector<uint64_t>, FrontierDecision,
                       FrontierMaskHash>* memo) {
    if (covered == full_mask) {
        return 0.0;
    }
    auto it = memo->find(covered);
    if (it != memo->end()) {
        return it->second.valid ? it->second.cost : kInfinity;
    }
    if (memo->size() >= memo_cap) {
        // Overflow: tell caller the DP didn't complete.  The caller falls
        // back to the positional DP.
        return kInfinity;
    }

    // Pick first uncovered topo-position whose predecessors are all covered.
    int v0_topo_idx = -1;
    for (size_t i = 0; i < n; ++i) {
        if (FrontierTestBit(covered, i)) continue;
        size_t op_id = graph.topo_order[i];
        bool ready = true;
        for (size_t pred : graph.preds[op_id]) {
            if (!FrontierTestBit(covered, graph.topo_pos[pred])) {
                ready = false;
                break;
            }
        }
        if (ready) {
            v0_topo_idx = static_cast<int>(i);
            break;
        }
    }
    if (v0_topo_idx < 0) {
        (*memo)[covered] = {};
        return kInfinity;
    }

    FrontierDecision best;
    if (static_cast<size_t>(v0_topo_idx) < groups_by_v0.size()) {
        for (size_t group_idx : groups_by_v0[v0_topo_idx]) {
            const FrontierGroup& fg = groups[group_idx];
            if (FrontierAnyOverlap(fg.mask, covered)) continue;

            // Every op in the group must be ready: its preds are either in
            // `covered` or internal to the group itself.
            bool group_ready = true;
            for (size_t i = 0; i < n && group_ready; ++i) {
                if (!FrontierTestBit(fg.mask, i)) continue;
                size_t op_id = graph.topo_order[i];
                for (size_t pred : graph.preds[op_id]) {
                    const size_t pt = graph.topo_pos[pred];
                    if (!FrontierTestBit(covered, pt) &&
                        !FrontierTestBit(fg.mask, pt)) {
                        group_ready = false;
                        break;
                    }
                }
            }
            if (!group_ready) continue;

            std::vector<uint64_t> new_covered = covered;
            FrontierOrInto(new_covered, fg.mask);
            const double group_cost = fg.candidate.metrics.latency;
            const double future = SolveFrontierDPImpl(
                problem, graph, groups, groups_by_v0, n,
                new_covered, full_mask, memo_cap, memo);
            if (!std::isfinite(future)) continue;
            const double total = group_cost + future;
            if (total < best.cost) {
                best.valid     = true;
                best.cost      = total;
                best.group_idx = group_idx;
            }
        }
    }

    (*memo)[covered] = best;
    return best.valid ? best.cost : kInfinity;
}

// Recover ordered schedule from the frontier DP memo.
static std::vector<CandidateGroup> RecoverFrontierSchedule(
    const std::vector<FrontierGroup>& groups,
    const std::vector<uint64_t>& full_mask,
    const std::unordered_map<std::vector<uint64_t>, FrontierDecision,
                             FrontierMaskHash>& memo) {
    std::vector<CandidateGroup> schedule;
    std::vector<uint64_t> covered(full_mask.size(), 0ULL);
    while (covered != full_mask) {
        auto it = memo.find(covered);
        if (it == memo.end() || !it->second.valid) break;
        const FrontierGroup& fg = groups[it->second.group_idx];
        schedule.push_back(fg.candidate);
        FrontierOrInto(covered, fg.mask);
    }
    return schedule;
}

// ============================================================================
// GRAPH-CUT DECOMPOSITION
// ============================================================================

// Returns sorted list of topo positions P where prefix_max_succ[P] == P.
// "prefix_max[i]" = max over j in [0..i] of max(topo_pos[succ] for succ in
// succs[topo_order[j]], and i itself).  When prefix_max[P] == P, op P is a
// DAG sink within [0..P]: it has no successors, so no edge crosses position P.
// Cut means: left segment [lo..P], right segment [P+1..N-1].
// No fusion candidate can span a natural cut → decomposition is lossless.
std::vector<size_t> FindNaturalCuts(const OpGraph& graph) {
    const size_t n = graph.topo_order.size();
    if (n <= 1) return {};
    std::vector<size_t> prefix_max(n, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t m = (i > 0) ? prefix_max[i - 1] : 0;
        const size_t op_id = graph.topo_order[i];
        for (size_t succ : graph.succs[op_id]) {
            m = std::max(m, graph.topo_pos[succ]);
        }
        prefix_max[i] = std::max(m, i);
    }
    std::vector<size_t> cuts;
    for (size_t p = 0; p + 1 < n; ++p) {
        if (prefix_max[p] == p) cuts.push_back(p);
    }
    return cuts;
}

// Returns a set of cut positions by scanning for the best cut opportunity
// within each chunk of max_seg_size ops (the "chunk cut" strategy).
//
// For each chunk boundary region, find the position P in a window around the
// ideal cut point that minimizes the number of "crossing edges" — i.e., edges
// from [0..P] to [P+1..N-1].  The position with the fewest crossing edges is
// chosen as the cut.  Ops at or near the cut with crossing edges are scheduled
// as singletons to avoid quality loss.
//
// This always produces cuts regardless of graph structure, enabling
// decomposition even for dense DAGs where strict natural cuts don't exist.
// The quality impact is bounded: a crossing edge means one op spans the cut
// boundary; if we schedule that op as a singleton, the solution still covers
// all ops.  The optimizer may miss fusing that op with its successor, but
// this is a controlled tradeoff for runtime.
std::vector<size_t> FindChunkCuts(const OpGraph& graph, size_t max_seg_size) {
    const size_t n = graph.topo_order.size();
    if (n <= max_seg_size) return {};

    // Precompute prefix_max_succ[i] = max topo_pos of any successor of any
    // op in [0..i] (or i itself if no successors).
    std::vector<size_t> prefix_max(n, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t m = (i > 0) ? prefix_max[i - 1] : 0;
        const size_t op_id = graph.topo_order[i];
        for (size_t succ : graph.succs[op_id]) {
            m = std::max(m, graph.topo_pos[succ]);
        }
        prefix_max[i] = std::max(m, i);
    }

    // Count crossing edges at each candidate cut position P:
    // crossing(P) = number of ops in [0..P] with at least one successor
    //               in [P+1..N-1].
    // = number of i in [0..P] where prefix_max(i's own successors) > P
    // We can compute this efficiently with a difference array.
    //
    // crossing(P) = number of i <= P where max_succ_topo[i] > P.
    // Precompute max_succ_topo[i] = max topo_pos among direct successors of
    // topo_order[i] (or i if no successors).
    std::vector<size_t> max_succ(n, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t m = i;  // default: no successor, contributes 0 crossing for cuts > i
        const size_t op_id = graph.topo_order[i];
        for (size_t succ : graph.succs[op_id]) {
            m = std::max(m, graph.topo_pos[succ]);
        }
        max_succ[i] = m;
    }

    // crossing(P) = #{i in [0..P] : max_succ[i] > P}
    // We find the best cut in each window [ideal-window/2 .. ideal+window/2].
    const size_t window = std::max<size_t>(4, max_seg_size / 4);
    std::vector<size_t> cuts;
    size_t seg_start = 0;

    while (seg_start + max_seg_size < n) {
        // Ideal cut: seg_start + max_seg_size - 1
        const size_t ideal = seg_start + max_seg_size - 1;
        const size_t search_lo = (ideal > window) ? (ideal - window) : seg_start;
        const size_t search_hi = std::min(ideal + window, n - 2);

        // Find P in [search_lo, search_hi] minimizing crossing(P).
        // Then minimize further by preferring P closer to ideal.
        size_t best_p = ideal;
        size_t best_cross = std::numeric_limits<size_t>::max();
        for (size_t p = search_lo; p <= search_hi; ++p) {
            // Count crossings for cut at P
            size_t cross = 0;
            for (size_t i = seg_start; i <= p; ++i) {
                if (max_succ[i] > p) ++cross;
            }
            if (cross < best_cross ||
                (cross == best_cross &&
                 (p > best_p ? p - ideal : ideal - p) <
                 (best_p > ideal ? best_p - ideal : ideal - best_p))) {
                best_cross = cross;
                best_p = p;
            }
        }
        cuts.push_back(best_p);
        seg_start = best_p + 1;
    }
    return cuts;
}

// Produces an OpGraph restricted to ops at topo positions [lo..hi] in
// full_graph, with topo positions remapped to [0..hi-lo].  Edges outside the
// segment are dropped.  Tensors produced outside the segment have
// producer_of_tensor[t] = -1 so ComputeBoundary treats them as graph inputs.
OpGraph ExtractSubgraph(const Problem& problem,
                        const OpGraph& full_graph,
                        size_t lo, size_t hi) {
    (void)problem;
    const size_t seg_n = hi - lo + 1;
    const size_t num_ops = full_graph.preds.size();
    OpGraph sub;

    // 1. Topo order (remapped indices 0..seg_n-1)
    sub.topo_order.resize(seg_n);
    for (size_t i = 0; i < seg_n; ++i)
        sub.topo_order[i] = full_graph.topo_order[lo + i];

    std::unordered_set<size_t> seg_set(sub.topo_order.begin(), sub.topo_order.end());

    // 2. topo_pos remapped; only segment op entries are meaningful
    sub.topo_pos.assign(num_ops, 0);
    for (size_t i = 0; i < seg_n; ++i)
        sub.topo_pos[sub.topo_order[i]] = i;

    // 3. preds / succs: intra-segment edges only
    sub.preds.assign(num_ops, {});
    sub.succs.assign(num_ops, {});
    for (size_t i = 0; i < seg_n; ++i) {
        const size_t op_id = sub.topo_order[i];
        for (size_t pred : full_graph.preds[op_id])
            if (seg_set.count(pred)) sub.preds[op_id].push_back(pred);
        for (size_t succ : full_graph.succs[op_id])
            if (seg_set.count(succ)) sub.succs[op_id].push_back(succ);
    }

    // 4. producer_of_tensor: -1 for tensors produced outside [lo..hi]
    sub.producer_of_tensor = full_graph.producer_of_tensor;
    for (size_t t = 0; t < sub.producer_of_tensor.size(); ++t) {
        const int prod = sub.producer_of_tensor[t];
        if (prod >= 0 && !seg_set.count(static_cast<size_t>(prod)))
            sub.producer_of_tensor[t] = -1;
    }

    // 5. consumers_of_tensor: segment ops only
    sub.consumers_of_tensor.assign(full_graph.consumers_of_tensor.size(), {});
    for (size_t t = 0; t < full_graph.consumers_of_tensor.size(); ++t)
        for (size_t c : full_graph.consumers_of_tensor[t])
            if (seg_set.count(c)) sub.consumers_of_tensor[t].push_back(c);

    return sub;
}

// Runs the appropriate DP for a sub-graph segment and returns an ordered
// schedule (vector of CandidateGroups).  Returns empty on failure; caller
// should fall back to the full-graph DP.
// Dispatches: bitmask DP for seg_n <= 20, frontier DP for seed-growth,
// positional DP as final fallback.
std::vector<CandidateGroup> SolveSegment(
    const Problem& problem,
    const OpGraph& sub_graph,
    const OptimusConfig& config) {

    const size_t seg_n = sub_graph.topo_order.size();
    if (seg_n == 0) return {};

    // Trivial single-op case
    if (seg_n == 1) {
        CandidateGroup cg = BuildBestCandidate(
            problem, sub_graph, size_t{0}, size_t{0}, config);
        if (cg.metrics.valid) return {std::move(cg)};
        return {};
    }

    // Generate candidates for this segment
    std::vector<std::vector<CandidateGroup>> seg_cands;
    if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth) {
        OptimusConfig seg_cfg = config;
        seg_cfg.allow_noncontig_groups = true;
        seg_cands = GenerateSeedGrowthCandidates(problem, sub_graph, seg_cfg);
    } else {
        seg_cands = GenerateCandidates(problem, sub_graph, config);
    }

    // Helper: add singleton fallbacks for any uncovered bitmask position
    auto add_bitmask_singletons = [&](std::vector<BitmaskGroup>& bgs) {
        std::unordered_set<int> covered_v0;
        for (const auto& bg : bgs) covered_v0.insert(bg.v0_topo_idx);
        for (size_t i = 0; i < seg_n; ++i) {
            if (covered_v0.count(static_cast<int>(i))) continue;
            CandidateGroup fb = BuildBestCandidate(problem, sub_graph, i, i, config);
            if (!fb.metrics.valid) continue;
            BitmaskGroup bg;
            bg.mask = (1ULL << i);
            bg.v0_topo_idx = static_cast<int>(i);
            bg.candidate = std::move(fb);
            bgs.push_back(std::move(bg));
        }
    };

    auto add_frontier_singletons = [&](std::vector<FrontierGroup>& fgs) {
        std::unordered_set<size_t> covered;
        for (const auto& fg : fgs) {
            size_t cnt = 0;
            for (uint64_t w : fg.mask) cnt += __builtin_popcountll(w);
            if (cnt == 1 &&
                FrontierTestBit(fg.mask, static_cast<size_t>(fg.v0_topo_idx)))
                covered.insert(static_cast<size_t>(fg.v0_topo_idx));
        }
        for (size_t i = 0; i < seg_n; ++i) {
            if (covered.count(i)) continue;
            CandidateGroup fb = BuildBestCandidate(problem, sub_graph, i, i, config);
            if (!fb.metrics.valid) continue;
            FrontierGroup fg;
            fg.mask.assign(FrontierNumWords(seg_n), 0ULL);
            FrontierSetBit(fg.mask, i);
            fg.v0_topo_idx = static_cast<int>(i);
            fg.candidate = std::move(fb);
            fgs.push_back(std::move(fg));
        }
    };

    // --- Bitmask DP (seg_n <= 20) ---
    if (seg_n <= 20) {
        std::vector<BitmaskGroup> bgs = BuildBitmaskGroups(sub_graph, seg_cands);
        add_bitmask_singletons(bgs);
        std::vector<std::vector<size_t>> by_v0(seg_n);
        for (size_t i = 0; i < bgs.size(); ++i) {
            const int v0 = bgs[i].v0_topo_idx;
            if (v0 >= 0 && static_cast<size_t>(v0) < seg_n) by_v0[v0].push_back(i);
        }
        const uint64_t full = (seg_n == 64u) ? ~0ULL : ((1ULL << seg_n) - 1u);
        std::unordered_map<uint64_t, BitmaskDecision> bm;
        const double cost = SolveBitmaskDPImpl(
            problem, sub_graph, bgs, by_v0, seg_n, 0, full, &bm);
        if (std::isfinite(cost)) {
            auto sched = RecoverBitmaskSchedule(bgs, full, bm);
            if (!sched.empty()) return sched;
        }
        // Fall through to positional DP

    } else if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth) {
        // --- Frontier DP (seg_n > 20, seed-growth) ---
        std::vector<FrontierGroup> fgs = BuildFrontierGroups(sub_graph, seg_cands);
        add_frontier_singletons(fgs);
        std::vector<std::vector<size_t>> by_v0(seg_n);
        for (size_t i = 0; i < fgs.size(); ++i) {
            const int v0 = fgs[i].v0_topo_idx;
            if (v0 >= 0 && static_cast<size_t>(v0) < seg_n) by_v0[v0].push_back(i);
        }
        std::vector<uint64_t> full_mask(FrontierNumWords(seg_n), 0ULL);
        for (size_t i = 0; i < seg_n; ++i) FrontierSetBit(full_mask, i);
        std::vector<uint64_t> zero(FrontierNumWords(seg_n), 0ULL);
        size_t memo_cap = 10000000;
        if (const char* e = std::getenv("MLSYS_OPTIMUS_FRONTIER_MEMO_CAP")) {
            const long v = std::atol(e);
            if (v > 0) memo_cap = static_cast<size_t>(v);
        }
        std::unordered_map<std::vector<uint64_t>, FrontierDecision,
                           FrontierMaskHash> fm;
        const double cost = SolveFrontierDPImpl(
            problem, sub_graph, fgs, by_v0, seg_n,
            zero, full_mask, memo_cap, &fm);
        if (std::isfinite(cost)) {
            auto sched = RecoverFrontierSchedule(fgs, full_mask, fm);
            if (!sched.empty()) return sched;
        }
        // Fall through to positional DP
    }

    // --- Positional DP fallback ---
    std::vector<std::unordered_set<size_t>> useful(seg_n);
    for (size_t s = 0; s < seg_n; ++s)
        for (const auto& c : seg_cands[s])
            useful[s].insert(c.boundary.boundary_inputs.begin(),
                              c.boundary.boundary_inputs.end());
    std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash> pm;
    (void)SolveFromState(problem, sub_graph, seg_cands, useful, 0, {}, &pm);

    // Recover positional schedule by walking the memo forward
    std::vector<CandidateGroup> pos_sched;
    size_t pos_start = 0;
    std::vector<size_t> retained;
    while (pos_start < seg_cands.size()) {
        const SearchStateKey key{
            pos_start, CanonicalizeRetainedInputs(retained, useful[pos_start])};
        auto it = pm.find(key);
        if (it == pm.end() || !it->second.valid ||
            it->second.candidate_index >= seg_cands[pos_start].size()) break;
        pos_sched.push_back(seg_cands[pos_start][it->second.candidate_index]);
        retained = it->second.retained_outputs;
        pos_start = pos_sched.back().end + 1;
    }
    return pos_sched;
}

}  // namespace

Solution SolveWithOptimusImpl(const Problem& problem, const OptimusConfig& config) {
    g_score_cache.clear();
    const OpGraph graph = BuildOpGraph(problem);
    std::vector<std::vector<CandidateGroup>> candidates;
    if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth) {
        std::cerr << "Optimus candidates: seed-growth\n";
        candidates = GenerateSeedGrowthCandidates(problem, graph, config);
    } else {
        std::cerr << "Optimus candidates: interval\n";
        candidates = GenerateCandidates(problem, graph, config);
    }
    if (config.guidance_mode == GuidanceMode::kConvAccelerator) {
        std::cerr << "Optimus guidance: conv_accelerator\n";
    }
    const size_t n = graph.topo_order.size();
    std::vector<std::unordered_set<size_t>> useful_inputs_by_start(n);
    for (size_t start = 0; start < n; ++start) {
        for (const auto& candidate : candidates[start]) {
            useful_inputs_by_start[start].insert(
                candidate.boundary.boundary_inputs.begin(),
                candidate.boundary.boundary_inputs.end());
        }
    }

    std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash> memo;
    (void)SolveFromState(problem, graph, candidates, useful_inputs_by_start, 0,
                         {}, &memo);
    return BuildSolutionFromSearch(problem, graph, candidates,
                                   useful_inputs_by_start, memo);
}

Solution SolveWithOptimus(const Problem& problem) {
    OptimusConfig config;
    config.candidate_mode = GetCandidateGenerationMode();
    config.guidance_mode = GuidanceMode::kContest;
    return SolveWithOptimusImpl(problem, config);
}

Solution SolveWithOptimusConvGuidance(const Problem& problem) {
    OptimusConfig config;
    config.candidate_mode = GetCandidateGenerationMode();
    config.guidance_mode = GuidanceMode::kConvAccelerator;
    config.conv_guidance_variant = ConvGuidanceVariant::kAdditivePenaltyV1;
    return SolveWithOptimusImpl(problem, config);
}

Solution SolveWithOptimusConvRerankV2(const Problem& problem) {
    OptimusConfig config;
    config.candidate_mode = GetCandidateGenerationMode();
    config.guidance_mode = GuidanceMode::kConvAccelerator;
    config.conv_guidance_variant = ConvGuidanceVariant::kLocalRerankV2;
    return SolveWithOptimusImpl(problem, config);
}

// Paper Algorithm 1 (adapted for latency minimization):
// - For N <= 20: bitmask DP over all precomputed candidate groups.
//   The bitmask state tracks which ops are covered, allowing the DP to find
//   optimal group assignments for branching DAGs that position-based DP might miss.
//   ISValid filters are applied when building the group list.
// - For N > 20: fall back to the existing position-based DP (also benefits
//   from the analytical tile candidates injected into BuildBestCandidate).
// ---------------------------------------------------------------------------
// Scheduler-driven solver: uses BuildBestCandidateWithScheduler for candidate
// generation and SolveFromStateWithScheduler for DP search.
// ---------------------------------------------------------------------------
Solution SolveWithOptimusImplWithScheduler(const Problem& problem,
                                           const OptimusConfig& config) {
    g_score_cache.clear();
    const OpGraph graph = BuildOpGraph(problem);

    // Generate candidates using the scheduler-extended builder.
    const size_t n = graph.topo_order.size();
    const size_t max_group_size = EstimateMaxGroupSize(problem);
    std::vector<std::vector<CandidateGroup>> candidates(n);

    if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth) {
        std::cerr << "Optimus sched candidates: seed-growth\n";
        // Seed-growth with scheduler builder.
        for (size_t topo_idx = 0; topo_idx < n; ++topo_idx) {
            const size_t seed = graph.topo_order[topo_idx];
            std::queue<std::vector<size_t>> pending;
            std::set<std::vector<size_t>> seen;
            pending.push({seed});
            seen.insert({seed});

            while (!pending.empty()) {
                std::vector<size_t> current_ops = pending.front();
                pending.pop();

                CandidateGroup candidate =
                    BuildBestCandidateWithScheduler(problem, graph, current_ops, config);
                if (candidate.metrics.valid &&
                    IsContiguousTopoSpan(graph, candidate)) {
                    if (candidate.ops.size() == 1 ||
                        candidate.internalized_bytes >=
                            std::max<int64_t>(1, problem.native_granularity.width) *
                                std::max<int64_t>(1, problem.native_granularity.height)) {
                        candidates[candidate.start].push_back(candidate);
                    }
                }

                if (current_ops.size() >= max_group_size) {
                    continue;
                }

                std::unordered_set<size_t> current_set(current_ops.begin(),
                                                        current_ops.end());
                std::unordered_set<size_t> frontier;
                for (size_t op_id : current_ops) {
                    for (size_t succ : graph.succs[op_id]) {
                        if (!current_set.count(succ)) {
                            frontier.insert(succ);
                        }
                    }
                }
                for (size_t next_op : frontier) {
                    std::vector<size_t> grown_ops = current_ops;
                    grown_ops.push_back(next_op);
                    std::sort(grown_ops.begin(), grown_ops.end(),
                              [&](size_t lhs, size_t rhs) {
                                  return graph.topo_pos[lhs] < graph.topo_pos[rhs];
                              });
                    if (seen.insert(grown_ops).second) {
                        pending.push(grown_ops);
                    }
                }
            }
        }

        // Deduplicate, rank, and cap per start.
        for (size_t start = 0; start < n; ++start) {
            auto& cands = candidates[start];
            std::sort(cands.begin(), cands.end(),
                      [](const CandidateGroup& lhs, const CandidateGroup& rhs) {
                          if (lhs.ops == rhs.ops) {
                              return lhs.metrics.latency < rhs.metrics.latency;
                          }
                          return lhs.ops < rhs.ops;
                      });
            cands.erase(
                std::unique(cands.begin(), cands.end(),
                            [](const CandidateGroup& lhs, const CandidateGroup& rhs) {
                                return lhs.ops == rhs.ops;
                            }),
                cands.end());
            std::sort(cands.begin(), cands.end(),
                      [](const CandidateGroup& lhs, const CandidateGroup& rhs) {
                          const double lhs_solo = static_cast<double>(lhs.ops.size());
                          const double rhs_solo = static_cast<double>(rhs.ops.size());
                          const double lhs_density = lhs.internalized_bytes -
                                                     lhs.metrics.latency / lhs_solo;
                          const double rhs_density = rhs.internalized_bytes -
                                                     rhs.metrics.latency / rhs_solo;
                          if (lhs_density != rhs_density) {
                              return lhs_density > rhs_density;
                          }
                          return lhs.metrics.latency < rhs.metrics.latency;
                      });
            if (cands.size() > 8) {
                cands.resize(8);
            }
            if (cands.empty()) {
                CandidateGroup fallback = BuildBestCandidateWithScheduler(
                    problem, graph, {graph.topo_order[start]}, config);
                if (fallback.metrics.valid) {
                    cands.push_back(std::move(fallback));
                }
            }
        }
    } else {
        std::cerr << "Optimus sched candidates: interval\n";
        // Interval-based with scheduler builder.
        for (size_t start = 0; start < n; ++start) {
            for (size_t end = start;
                 end < n && end < start + max_group_size; ++end) {
                CandidateGroup candidate =
                    BuildBestCandidateWithScheduler(problem, graph, start, end, config);
                if (!candidate.metrics.valid) {
                    continue;
                }
                if (candidate.ops.size() > 1) {
                    const int64_t min_gain =
                        std::max<int64_t>(1, problem.native_granularity.width) *
                        std::max<int64_t>(1, problem.native_granularity.height);
                    if (candidate.internalized_bytes < min_gain) {
                        continue;
                    }
                }
                candidates[start].push_back(std::move(candidate));
            }
            if (candidates[start].empty()) {
                CandidateGroup fallback = BuildBestCandidateWithScheduler(
                    problem, graph, start, start, config);
                if (fallback.metrics.valid) {
                    candidates[start].push_back(std::move(fallback));
                }
            }
        }
    }

    std::cerr << "Optimus sched: weight=" << config.scheduler_cost_weight
              << " propose=" << (config.scheduler_propose_granularity ? "yes" : "no")
              << "\n";

    // Build useful inputs map.
    std::vector<std::unordered_set<size_t>> useful_inputs_by_start(n);
    for (size_t start = 0; start < n; ++start) {
        for (const auto& candidate : candidates[start]) {
            useful_inputs_by_start[start].insert(
                candidate.boundary.boundary_inputs.begin(),
                candidate.boundary.boundary_inputs.end());
        }
    }

    // Run scheduler-aware DP.
    std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash> memo;
    (void)SolveFromStateWithScheduler(problem, graph, candidates,
                                       useful_inputs_by_start, 0, {}, config,
                                       &memo);

    // Build solution (uses same reconstruction — evaluator is ground truth).
    return BuildSolutionFromSearch(problem, graph, candidates,
                                   useful_inputs_by_start, memo);
}

Solution SolveWithOptimusSched(const Problem& problem) {
    OptimusConfig config;
    config.candidate_mode = GetCandidateGenerationMode();
    config.guidance_mode = GuidanceMode::kConvAccelerator;
    config.conv_guidance_variant = ConvGuidanceVariant::kLocalRerankV2;

    // Parse scheduler-specific env vars.
    const char* weight_raw = std::getenv("MLSYS_SCHED_WEIGHT");
    if (weight_raw != nullptr) {
        config.scheduler_cost_weight = std::atof(weight_raw);
    }

    const char* propose_raw = std::getenv("MLSYS_SCHED_PROPOSE");
    if (propose_raw != nullptr) {
        std::string propose_str(propose_raw);
        std::transform(propose_str.begin(), propose_str.end(), propose_str.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        config.scheduler_propose_granularity =
            (propose_str != "false" && propose_str != "0" && propose_str != "no");
    } else {
        config.scheduler_propose_granularity = true;  // Default: enabled.
    }

    return SolveWithOptimusImplWithScheduler(problem, config);
}

Solution SolveWithPaperOptimus(const Problem& problem) {
    g_score_cache.clear();
    const OpGraph graph = BuildOpGraph(problem);
    const size_t n = graph.topo_order.size();

    std::cerr << "Optimus paper: N=" << n << "\n";

    OptimusConfig config;
    config.candidate_mode = GetCandidateGenerationMode();
    config.guidance_mode = GuidanceMode::kContest;
    if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth)
        config.allow_noncontig_groups = true;

    // === Graph-cut decomposition: split at natural or chunk cuts ===
    // Detect cuts BEFORE generating candidates: if we take the decomposition
    // path each segment generates its own candidates (cheaper, local), so we
    // avoid wasting time on full-graph candidate generation.
    // Strategy 1: Natural cuts (lossless — no edge crosses the cut).
    // Strategy 2: Chunk cuts at positions with fewest crossing edges (heuristic).
    // Both strategies fall back to full DP if any segment fails.
    if (n > 20) {
        // Determine max segment size: aim for ~25 ops per segment so each
        // segment uses the fast bitmask DP (O(2^seg_n)).  Overridable via env.
        size_t max_seg = 25;
        if (const char* e = std::getenv("MLSYS_OPTIMUS_MAX_SEG_SIZE")) {
            const long v = std::atol(e);
            if (v > 0) max_seg = static_cast<size_t>(v);
        }

        std::vector<size_t> cuts = FindNaturalCuts(graph);
        const char* cut_type = "natural";
        if (cuts.empty()) {
            cuts = FindChunkCuts(graph, max_seg);
            cut_type = "chunk";
        }

        if (!cuts.empty()) {
            std::cerr << "Optimus paper: found " << cuts.size()
                      << " " << cut_type << " cuts\n";

            // Build segments: [0..c0], [c0+1..c1], ..., [c_{k-1}+1..n-1]
            std::vector<std::pair<size_t, size_t>> segments;
            size_t prev = 0;
            for (size_t cut : cuts) {
                segments.push_back({prev, cut});
                prev = cut + 1;
            }
            segments.push_back({prev, n - 1});

            for (size_t i = 0; i < segments.size(); ++i) {
                std::cerr << "Optimus paper: segment " << i
                          << " size="
                          << (segments[i].second - segments[i].first + 1)
                          << " topo=[" << segments[i].first
                          << ".." << segments[i].second << "]\n";
            }

            std::vector<CandidateGroup> full_sched;
            bool decomp_ok = true;
            for (const auto& [lo, hi] : segments) {
                OpGraph sub = ExtractSubgraph(problem, graph, lo, hi);
                auto seg_sched = SolveSegment(problem, sub, config);
                if (seg_sched.empty()) {
                    std::cerr << "Optimus paper: segment [" << lo << ".."
                              << hi << "] solve failed, falling back to "
                                 "full DP\n";
                    decomp_ok = false;
                    break;
                }
                for (auto& cg : seg_sched)
                    full_sched.push_back(std::move(cg));
            }
            if (decomp_ok && !full_sched.empty()) {
                Solution sol = BuildSolutionFromSchedule(problem, full_sched);
                if (!sol.subgraphs.empty()) {
                    std::cerr << "Optimus paper: decomposition succeeded ("
                              << sol.subgraphs.size() << " subgraphs)\n";
                    return sol;
                }
            }
            std::cerr << "Optimus paper: decomposition fallback to full DP\n";
        }
    }
    // === End graph-cut decomposition ===

    // Generate candidates for the full-graph DP path.
    // This runs only when: N<=20 (no decomposition attempted), no cuts found,
    // or decomposition failed and we fell through to here.
    std::vector<std::vector<CandidateGroup>> candidates;
    if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth) {
        std::cerr << "Optimus paper: seed-growth candidates (non-contig allowed)\n";
        candidates = GenerateSeedGrowthCandidates(problem, graph, config);
    } else {
        candidates = GenerateCandidates(problem, graph, config);
    }

    if (n > 20) {
        // For N > 20, prefer the frontier set-cover DP when seed-growth is
        // active (needed to evaluate non-contiguous candidates).  Fall back
        // to the positional DP on memo-cap overflow or empty schedule.
        if (config.candidate_mode == CandidateGenerationMode::kSeedGrowth) {
            std::cerr << "Optimus paper: frontier DP (N=" << n << ")\n";
            std::vector<FrontierGroup> fgroups =
                BuildFrontierGroups(graph, candidates);
            std::cerr << "Optimus paper: " << fgroups.size()
                      << " frontier groups\n";

            // Ensure every topo position has at least one singleton group
            // whose mask is exactly that single position. Seed-growth only
            // guarantees v0_topo_idx coverage, which is necessary but not
            // sufficient: a multi-op group has v0 equal to its smallest
            // position but cannot be picked when earlier positions in that
            // group's topo-prefix-range are uncovered.
            {
                std::unordered_set<size_t> singleton_positions;
                for (const auto& fg : fgroups) {
                    // Count bits set; if exactly 1 and it equals v0, this is
                    // a singleton for that position.
                    size_t bit_count = 0;
                    for (uint64_t w : fg.mask) bit_count += __builtin_popcountll(w);
                    if (bit_count == 1 &&
                        FrontierTestBit(fg.mask,
                                        static_cast<size_t>(fg.v0_topo_idx))) {
                        singleton_positions.insert(
                            static_cast<size_t>(fg.v0_topo_idx));
                    }
                }
                size_t added = 0, invalid = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (singleton_positions.count(i)) continue;
                    CandidateGroup fallback = BuildBestCandidate(
                        problem, graph, i, i, config);
                    if (!fallback.metrics.valid) { ++invalid; continue; }
                    FrontierGroup fg;
                    fg.mask.assign(FrontierNumWords(n), 0ULL);
                    FrontierSetBit(fg.mask, i);
                    fg.v0_topo_idx = static_cast<int>(i);
                    fg.candidate = std::move(fallback);
                    fgroups.push_back(std::move(fg));
                    ++added;
                }
                std::cerr << "Optimus paper: added " << added
                          << " fallback singletons (" << invalid
                          << " invalid, "
                          << singleton_positions.size()
                          << " already covered)\n";
            }

            std::vector<std::vector<size_t>> groups_by_v0(n);
            for (size_t i = 0; i < fgroups.size(); ++i) {
                int v0 = fgroups[i].v0_topo_idx;
                if (v0 >= 0 && static_cast<size_t>(v0) < n) {
                    groups_by_v0[v0].push_back(i);
                }
            }
            // Diagnostic: flag topo positions with no groups.  Every position
            // must have at least the singleton (added above) unless its
            // BuildBestCandidate returned invalid.  Missing coverage => the
            // frontier DP cannot make forward progress past that position.
            if (LegalityDebugEnabled()) {
                size_t missing = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (groups_by_v0[i].empty()) {
                        if (missing < 20) {
                            std::cerr << "Optimus paper: no groups for topo pos "
                                      << i << " (op "
                                      << graph.topo_order[i] << ")\n";
                        }
                        ++missing;
                    }
                }
                if (missing > 0) {
                    std::cerr << "Optimus paper: " << missing
                              << " topo positions have no groups\n";
                }
            }

            std::vector<uint64_t> full_mask(FrontierNumWords(n), 0ULL);
            for (size_t i = 0; i < n; ++i) FrontierSetBit(full_mask, i);
            std::vector<uint64_t> covered(FrontierNumWords(n), 0ULL);

            // Default cap sized so bm-17 (N=103, ~5M reachable states in
            // practice) completes without truncation. Overridable via env.
            size_t memo_cap = 10000000;
            if (const char* cap_env = std::getenv("MLSYS_OPTIMUS_FRONTIER_MEMO_CAP")) {
                const long parsed = std::atol(cap_env);
                if (parsed > 0) memo_cap = static_cast<size_t>(parsed);
            }

            std::unordered_map<std::vector<uint64_t>, FrontierDecision,
                               FrontierMaskHash> fmemo;
            const double fcost = SolveFrontierDPImpl(
                problem, graph, fgroups, groups_by_v0, n,
                covered, full_mask, memo_cap, &fmemo);
            // Count how many memoized states are dead-ends vs leads.
            size_t dead_states = 0, live_states = 0;
            for (const auto& kv : fmemo) {
                if (kv.second.valid) ++live_states; else ++dead_states;
            }
            std::cerr << "Optimus paper frontier DP cost: " << fcost
                      << " memo_size=" << fmemo.size()
                      << " live=" << live_states
                      << " dead=" << dead_states
                      << " cap=" << memo_cap << "\n";

            if (std::isfinite(fcost)) {
                auto schedule = RecoverFrontierSchedule(fgroups, full_mask, fmemo);
                if (!schedule.empty()) {
                    Solution sol = BuildSolutionFromSchedule(problem, schedule);
                    if (!sol.subgraphs.empty()) return sol;
                }
            }
            std::cerr << "Optimus paper: frontier DP unusable, "
                         "falling back to positional DP\n";
        }

        // Positional DP fallback (interval candidates, or frontier overflow).
        std::cerr << "Optimus paper: positional DP fallback (N=" << n << ")\n";
        const size_t num_ops = graph.topo_order.size();
        std::vector<std::unordered_set<size_t>> useful_inputs_by_start(num_ops);
        for (size_t start = 0; start < num_ops; ++start) {
            for (const auto& cand : candidates[start]) {
                useful_inputs_by_start[start].insert(
                    cand.boundary.boundary_inputs.begin(),
                    cand.boundary.boundary_inputs.end());
            }
        }
        std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash> memo;
        (void)SolveFromState(problem, graph, candidates, useful_inputs_by_start, 0,
                             {}, &memo);
        return BuildSolutionFromSearch(problem, graph, candidates,
                                       useful_inputs_by_start, memo);
    }

    // N <= 20: build bitmask groups (with ISValid filter) and run bitmask DP.
    std::vector<BitmaskGroup> bitmask_groups = BuildBitmaskGroups(graph, candidates);
    std::cerr << "Optimus paper: " << bitmask_groups.size() << " bitmask groups\n";

    // Ensure every op has at least a single-op fallback group.
    {
        std::unordered_set<int> covered_v0;
        for (const auto& bg : bitmask_groups) {
            covered_v0.insert(bg.v0_topo_idx);
        }
        for (size_t i = 0; i < n; ++i) {
            if (!covered_v0.count(static_cast<int>(i))) {
                // Build a single-op candidate for this op
                CandidateGroup fallback = BuildBestCandidate(
                    problem, graph, i, i, config);
                if (fallback.metrics.valid) {
                    BitmaskGroup bg;
                    bg.mask = (1ULL << i);
                    bg.v0_topo_idx = static_cast<int>(i);
                    bg.candidate = std::move(fallback);
                    bitmask_groups.push_back(std::move(bg));
                }
            }
        }
    }

    // Index groups by v0_topo_idx for O(1) lookup during DP.
    std::vector<std::vector<size_t>> groups_by_v0(n);
    for (size_t i = 0; i < bitmask_groups.size(); ++i) {
        int v0 = bitmask_groups[i].v0_topo_idx;
        if (v0 >= 0 && static_cast<size_t>(v0) < n) {
            groups_by_v0[v0].push_back(i);
        }
    }

    const uint64_t full_mask = (n == 64u) ? ~0ULL : ((1ULL << n) - 1u);
    std::unordered_map<uint64_t, BitmaskDecision> memo;
    const double best_cost = SolveBitmaskDPImpl(
        problem, graph, bitmask_groups, groups_by_v0, n, 0, full_mask, &memo);
    std::cerr << "Optimus paper bitmask DP cost: " << best_cost << "\n";

    auto schedule = RecoverBitmaskSchedule(bitmask_groups, full_mask, memo);

    if (schedule.empty()) {
        std::cerr << "Optimus paper: bitmask DP produced empty schedule, falling back\n";
        const size_t num_ops = graph.topo_order.size();
        std::vector<std::unordered_set<size_t>> useful_inputs_by_start(num_ops);
        for (size_t start = 0; start < num_ops; ++start) {
            for (const auto& cand : candidates[start]) {
                useful_inputs_by_start[start].insert(
                    cand.boundary.boundary_inputs.begin(),
                    cand.boundary.boundary_inputs.end());
            }
        }
        std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash> memo2;
        (void)SolveFromState(problem, graph, candidates, useful_inputs_by_start, 0,
                             {}, &memo2);
        return BuildSolutionFromSearch(problem, graph, candidates,
                                       useful_inputs_by_start, memo2);
    }

    Solution sol = BuildSolutionFromSchedule(problem, schedule);
    if (sol.subgraphs.empty()) {
        std::cerr << "Optimus paper: schedule build failed, falling back\n";
        const size_t num_ops = graph.topo_order.size();
        std::vector<std::unordered_set<size_t>> useful_inputs_by_start(num_ops);
        for (size_t start = 0; start < num_ops; ++start) {
            for (const auto& cand : candidates[start]) {
                useful_inputs_by_start[start].insert(
                    cand.boundary.boundary_inputs.begin(),
                    cand.boundary.boundary_inputs.end());
            }
        }
        std::unordered_map<SearchStateKey, SearchDecision, SearchStateKeyHash> memo2;
        (void)SolveFromState(problem, graph, candidates, useful_inputs_by_start, 0,
                             {}, &memo2);
        return BuildSolutionFromSearch(problem, graph, candidates,
                                       useful_inputs_by_start, memo2);
    }
    return sol;
}

}  // namespace mlsys
