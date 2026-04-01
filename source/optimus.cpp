#include "optimus.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
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

struct StepEstimate {
    int64_t active_bytes = 0;
    int64_t output_write_bytes = 0;
    double compute_cost = 0.0;
};

struct CandidateGroup {
    size_t start = 0;
    size_t end = 0;
    std::vector<size_t> ops;
    GroupBoundary boundary;
    Granularity granularity{};
    GroupMetrics metrics;
    int64_t internalized_bytes = 0;
};

struct DpState {
    double cost = std::numeric_limits<double>::infinity();
    size_t next_index = 0;
    size_t candidate_index = 0;
};

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr size_t kDefaultMaxGroupSize = 4;

int64_t CeilDivInt(int64_t value, int64_t divisor) {
    if (divisor <= 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

int64_t TensorElements(const Tensor& tensor) {
    return tensor.width * tensor.height;
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

std::vector<int64_t> BuildSpatialCandidates(int64_t dim, int64_t native_dim) {
    std::set<int64_t> values;
    values.insert(std::max<int64_t>(1, dim));
    values.insert(std::max<int64_t>(1, std::min(dim, native_dim)));
    values.insert(std::max<int64_t>(1, std::min(dim, native_dim / 2)));
    values.insert(std::max<int64_t>(1, std::min(dim, native_dim / 4)));

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
    values.insert(std::max<int64_t>(1, std::min(max_k, native_k)));
    values.insert(std::max<int64_t>(1, std::min(max_k, native_k / 2)));
    values.insert(std::max<int64_t>(1, std::min(max_k, native_k / 4)));

    std::vector<int64_t> result(values.begin(), values.end());
    std::sort(result.begin(), result.end(), std::greater<int64_t>());
    return result;
}

int64_t EstimateWorkingSetBytes(const Problem& problem, const OpGraph& graph,
                                const std::vector<size_t>& ops,
                                const GroupBoundary& boundary,
                                const Granularity& granularity);

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

    int64_t peak = 0;
    for (int64_t row_tile = 0; row_tile < tiles_h; ++row_tile) {
        for (int64_t col_tile = 0; col_tile < tiles_w; ++col_tile) {
            const int64_t actual_h =
                std::min<int64_t>(granularity.height,
                                  ref_out.height - row_tile * granularity.height);
            const int64_t actual_w =
                std::min<int64_t>(granularity.width,
                                  ref_out.width - col_tile * granularity.width);
            for (int64_t step_index = 0; step_index < max_k_steps; ++step_index) {
                peak = std::max(peak, EstimateStep(problem, graph, ops, boundary,
                                                   granularity, actual_h, actual_w,
                                                   step_index, max_k_steps)
                                          .active_bytes);
            }
        }
    }

    return peak;
}

double EstimateLatency(const Problem& problem, const OpGraph& graph,
                       const std::vector<size_t>& ops,
                       const GroupBoundary& boundary,
                       const Granularity& granularity,
                       const std::unordered_set<size_t>& retained_inputs = {},
                       const std::unordered_set<size_t>& retained_outputs = {}) {
    if (ops.empty()) {
        return kInfinity;
    }

    const auto& ref_out =
        problem.tensors[problem.ops[ops.front()].outputs.front()];
    const int64_t tile_w = std::max<int64_t>(1, granularity.width);
    const int64_t tile_h = std::max<int64_t>(1, granularity.height);
    const int64_t tile_k = std::max<int64_t>(1, granularity.depth);
    const int64_t native_w = std::max<int64_t>(1, problem.native_granularity.width);
    const int64_t spatial_tiles =
        CeilDivInt(ref_out.width, tile_w) * CeilDivInt(ref_out.height, tile_h);
    (void)native_w;

    int64_t max_k_steps = 1;
    for (size_t op_id : ops) {
        if (IsMatMul(problem.ops[op_id])) {
            max_k_steps = std::max(max_k_steps,
                                   CeilDivInt(MatMulK(problem, op_id), tile_k));
        }
    }

    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    double total_latency = 0.0;

    for (int64_t tile = 0; tile < spatial_tiles; ++tile) {
        const int64_t row_tile = tile / CeilDivInt(ref_out.width, tile_w);
        const int64_t col_tile = tile % CeilDivInt(ref_out.width, tile_w);
        const int64_t actual_h =
            std::min<int64_t>(tile_h, ref_out.height - row_tile * tile_h);
        const int64_t actual_w =
            std::min<int64_t>(tile_w, ref_out.width - col_tile * tile_w);
        for (int64_t step = 0; step < max_k_steps; ++step) {
            StepEstimate estimate = EstimateStep(problem, graph, ops, boundary,
                                                 granularity, actual_h, actual_w,
                                                 step, max_k_steps);
            double memory_in = static_cast<double>(estimate.active_bytes);
            for (size_t tensor_id : boundary.boundary_inputs) {
                if (retained_inputs.count(tensor_id)) {
                    const auto& tensor = problem.tensors[tensor_id];
                    if (IsPointwise(problem.ops[ops.front()])) {
                        memory_in -= static_cast<double>(actual_h * actual_w);
                    } else {
                        (void)tensor;
                    }
                }
            }
            // Recompute the memory-in term with retain-awareness using the exact
            // per-op input accounting logic.
            memory_in = 0.0;
            const int64_t k_begin = step * tile_k;
            const bool final_k_step = (step == max_k_steps - 1);
            for (size_t op_id : ops) {
                const auto& op = problem.ops[op_id];
                if (IsPointwise(op)) {
                    if (!final_k_step) {
                        continue;
                    }
                    for (size_t tensor_id : op.inputs) {
                        int producer = graph.producer_of_tensor[tensor_id];
                        if (producer >= 0 &&
                            op_set.count(static_cast<size_t>(producer))) {
                            continue;
                        }
                        if (!retained_inputs.count(tensor_id)) {
                            memory_in += static_cast<double>(actual_h * actual_w);
                        }
                    }
                } else if (IsMatMul(op)) {
                    const int64_t k_total = MatMulK(problem, op_id);
                    if (k_begin >= k_total) {
                        continue;
                    }
                    const int64_t k_step =
                        std::min<int64_t>(tile_k, k_total - k_begin);
                    for (size_t input_index = 0; input_index < op.inputs.size();
                         ++input_index) {
                        size_t tensor_id = op.inputs[input_index];
                        int producer = graph.producer_of_tensor[tensor_id];
                        if (producer >= 0 &&
                            op_set.count(static_cast<size_t>(producer))) {
                            continue;
                        }
                        if (retained_inputs.count(tensor_id)) {
                            continue;
                        }
                        if (input_index == 0) {
                            memory_in += static_cast<double>(actual_h * k_step);
                        } else {
                            memory_in += static_cast<double>(actual_w * k_step);
                        }
                    }
                }
            }

            double memory_out = 0.0;
            if (final_k_step) {
                for (size_t tensor_id : boundary.boundary_outputs) {
                    if (!retained_outputs.count(tensor_id)) {
                        memory_out += static_cast<double>(actual_h * actual_w);
                    }
                }
            }

            total_latency += std::max(estimate.compute_cost,
                         (memory_in + memory_out) /
                             static_cast<double>(problem.slow_memory_bandwidth));
        }
    }

    return total_latency;
}

GroupMetrics EvaluateGroup(const Problem& problem, const OpGraph& graph,
                           const std::vector<size_t>& ops,
                           const GroupBoundary& boundary,
                           const Granularity& granularity) {
    GroupMetrics metrics;
    metrics.working_set_bytes =
        EstimateWorkingSetBytes(problem, graph, ops, boundary, granularity);
    if (metrics.working_set_bytes > problem.fast_memory_capacity) {
        return metrics;
    }

    metrics.latency =
        EstimateLatency(problem, graph, ops, boundary, granularity);
    metrics.valid = std::isfinite(metrics.latency);
    return metrics;
}

CandidateGroup BuildBestCandidate(const Problem& problem, const OpGraph& graph,
                                  size_t start, size_t end) {
    CandidateGroup candidate;
    candidate.start = start;
    candidate.end = end;
    for (size_t i = start; i <= end; ++i) {
        candidate.ops.push_back(graph.topo_order[i]);
    }

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

    const auto& ref_out =
        problem.tensors[problem.ops[candidate.ops.front()].outputs.front()];
    const auto width_candidates =
        BuildSpatialCandidates(ref_out.width, problem.native_granularity.width);
    const auto height_candidates =
        BuildSpatialCandidates(ref_out.height, problem.native_granularity.height);
    const auto k_candidates = BuildKCandidates(problem, candidate.ops);

    for (int64_t width : width_candidates) {
        for (int64_t height : height_candidates) {
            for (int64_t depth : k_candidates) {
                Granularity granularity{width, height, depth};
                GroupMetrics metrics = EvaluateGroup(
                    problem, graph, candidate.ops, candidate.boundary, granularity);
                if (!metrics.valid) {
                    continue;
                }
                if (!candidate.metrics.valid || metrics.latency < candidate.metrics.latency) {
                    candidate.granularity = granularity;
                    candidate.metrics = metrics;
                }
            }
        }
    }

    return candidate;
}

size_t EstimateMaxGroupSize(const Problem& problem) {
    const int64_t native_tile =
        std::max<int64_t>(1, problem.native_granularity.width) *
        std::max<int64_t>(1, problem.native_granularity.height);
    if (native_tile <= 0) {
        return 2;
    }
    const int64_t ratio = problem.fast_memory_capacity / native_tile;
    if (ratio <= 2) {
        return 2;
    }
    if (ratio <= 8) {
        return 3;
    }
    return kDefaultMaxGroupSize;
}

std::vector<std::vector<CandidateGroup>> GenerateCandidates(
    const Problem& problem, const OpGraph& graph) {
    const size_t n = graph.topo_order.size();
    const size_t max_group_size = EstimateMaxGroupSize(problem);
    std::vector<std::vector<CandidateGroup>> by_start(n);

    for (size_t start = 0; start < n; ++start) {
        for (size_t end = start;
             end < n && end < start + max_group_size; ++end) {
            CandidateGroup candidate =
                BuildBestCandidate(problem, graph, start, end);
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

            by_start[start].push_back(std::move(candidate));
        }

        if (by_start[start].empty()) {
            CandidateGroup fallback =
                BuildBestCandidate(problem, graph, start, start);
            if (fallback.metrics.valid) {
                by_start[start].push_back(std::move(fallback));
            }
        }
    }

    return by_start;
}

std::vector<size_t> ChooseRetainedOutputs(
    const Problem& problem, const CandidateGroup& current,
    const CandidateGroup& next,
    const std::unordered_set<size_t>& retained_prev) {
    std::unordered_set<size_t> next_inputs(next.boundary.boundary_inputs.begin(),
                                           next.boundary.boundary_inputs.end());
    std::vector<size_t> candidates;
    for (size_t tensor_id : current.boundary.boundary_outputs) {
        if (next_inputs.count(tensor_id)) {
            candidates.push_back(tensor_id);
        }
    }
    if (candidates.empty()) {
        return {};
    }

    int64_t retained_prev_bytes = 0;
    for (size_t tensor_id : current.boundary.boundary_inputs) {
        if (retained_prev.count(tensor_id)) {
            retained_prev_bytes += TensorElements(problem.tensors[tensor_id]);
        }
    }

    const int64_t next_limit =
        problem.fast_memory_capacity - next.metrics.working_set_bytes;
    const int64_t current_limit = problem.fast_memory_capacity -
                                  current.metrics.working_set_bytes -
                                  retained_prev_bytes;
    const int64_t available_bytes = std::min(next_limit, current_limit);
    if (available_bytes <= 0) {
        return {};
    }

    if (candidates.size() <= 12) {
        std::vector<size_t> best;
        int64_t best_bytes = 0;
        const size_t total = static_cast<size_t>(1) << candidates.size();
        for (size_t mask = 1; mask < total; ++mask) {
            int64_t bytes = 0;
            std::vector<size_t> chosen;
            for (size_t i = 0; i < candidates.size(); ++i) {
                if ((mask >> i) & 1U) {
                    bytes += TensorElements(problem.tensors[candidates[i]]);
                    chosen.push_back(candidates[i]);
                }
            }
            if (bytes <= available_bytes && bytes > best_bytes) {
                best_bytes = bytes;
                best = std::move(chosen);
            }
        }
        std::sort(best.begin(), best.end());
        return best;
    }

    std::sort(candidates.begin(), candidates.end(),
              [&](size_t lhs, size_t rhs) {
                  return TensorElements(problem.tensors[lhs]) >
                         TensorElements(problem.tensors[rhs]);
              });
    std::vector<size_t> chosen;
    int64_t used_bytes = 0;
    for (size_t tensor_id : candidates) {
        int64_t tensor_bytes = TensorElements(problem.tensors[tensor_id]);
        if (used_bytes + tensor_bytes <= available_bytes) {
            used_bytes += tensor_bytes;
            chosen.push_back(tensor_id);
        }
    }
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

Solution RecomputeScheduleLatencies(const Problem& problem, const OpGraph& graph,
                                    const std::vector<CandidateGroup>& schedule) {
    Solution solution;
    if (schedule.empty()) {
        return solution;
    }

    std::vector<std::vector<size_t>> retained(schedule.size());
    std::unordered_set<size_t> retained_prev;
    for (size_t i = 0; i + 1 < schedule.size(); ++i) {
        retained[i] = ChooseRetainedOutputs(problem, schedule[i], schedule[i + 1],
                                            retained_prev);
        retained_prev.clear();
        retained_prev.insert(retained[i].begin(), retained[i].end());
    }

    for (size_t i = 0; i < schedule.size(); ++i) {
        std::unordered_set<size_t> retained_inputs;
        if (i > 0) {
            retained_inputs.insert(retained[i - 1].begin(), retained[i - 1].end());
        }
        std::unordered_set<size_t> retained_outputs(retained[i].begin(),
                                                    retained[i].end());

        Subgraph subgraph;
        subgraph.op_ids = schedule[i].ops;
        subgraph.granularity = schedule[i].granularity;
        subgraph.tensors_to_retain = retained[i];
        subgraph.traversal_order = std::nullopt;
        subgraph.subgraph_latency = EstimateLatency(
            problem, graph, schedule[i].ops, schedule[i].boundary,
            schedule[i].granularity, retained_inputs, retained_outputs);
        solution.subgraphs.push_back(std::move(subgraph));
    }

    return solution;
}

}  // namespace

Solution SolveWithOptimus(const Problem& problem) {
    const OpGraph graph = BuildOpGraph(problem);
    const auto candidates = GenerateCandidates(problem, graph);
    const size_t n = graph.topo_order.size();

    std::vector<DpState> dp(n + 1);
    dp[n].cost = 0.0;
    dp[n].next_index = n;
    dp[n].candidate_index = 0;

    for (size_t i = n; i-- > 0;) {
        for (size_t cand_idx = 0; cand_idx < candidates[i].size(); ++cand_idx) {
            const auto& candidate = candidates[i][cand_idx];
            const size_t next = candidate.end + 1;
            if (!std::isfinite(dp[next].cost)) {
                continue;
            }
            const double cost = candidate.metrics.latency + dp[next].cost;
            if (cost < dp[i].cost) {
                dp[i].cost = cost;
                dp[i].next_index = next;
                dp[i].candidate_index = cand_idx;
            }
        }
    }

    std::vector<CandidateGroup> schedule;
    for (size_t i = 0; i < n;) {
        if (candidates[i].empty() || !std::isfinite(dp[i].cost)) {
            CandidateGroup fallback = BuildBestCandidate(problem, graph, i, i);
            schedule.push_back(std::move(fallback));
            ++i;
            continue;
        }
        const auto& chosen = candidates[i][dp[i].candidate_index];
        schedule.push_back(chosen);
        i = dp[i].next_index;
    }

    return RecomputeScheduleLatencies(problem, graph, schedule);
}

}  // namespace mlsys
