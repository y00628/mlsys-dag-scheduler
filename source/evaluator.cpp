#include "evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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

struct BoundaryInfo {
    std::vector<size_t> boundary_inputs;
    std::vector<size_t> boundary_outputs;
    std::vector<size_t> internal_tensors;
};

struct ResidentSlice {
    int64_t bytes = 0;
    int64_t last_used = 0;
};

struct SliceKey {
    size_t tensor_id = 0;
    int kind = 0;
    int64_t row = 0;
    int64_t col = 0;
    int64_t k_begin = 0;

    bool operator==(const SliceKey& other) const {
        return tensor_id == other.tensor_id && kind == other.kind &&
               row == other.row && col == other.col && k_begin == other.k_begin;
    }
};

struct SliceKeyHash {
    size_t operator()(const SliceKey& key) const {
        size_t h = std::hash<size_t>{}(key.tensor_id);
        h ^= std::hash<int>{}(key.kind) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.row) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.col) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k_begin) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct StepRequirement {
    std::unordered_map<SliceKey, int64_t, SliceKeyHash> slices;
    double compute_cost = 0.0;
    int64_t output_active_bytes = 0;
    int64_t output_write_bytes = 0;
    std::vector<size_t> retained_outputs_completed;
};

int64_t CeilDiv(int64_t value, int64_t divisor) {
    if (divisor <= 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

int64_t TensorBytes(const Tensor& tensor) {
    return tensor.width * tensor.height;
}

bool IsMatMul(const Op& op) {
    return op.op_type == "MatMul";
}

bool IsPointwise(const Op& op) {
    return op.op_type == "Pointwise";
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

bool IsUnaryPointwiseOp(const Problem& problem, size_t op_id) {
    const auto& op = problem.ops[op_id];
    return IsPointwise(op) && op.inputs.size() == 1 && op.outputs.size() == 1;
}

bool IsPreOpFusionChain(const Problem& problem, const OpGraph& graph,
                        const std::vector<size_t>& ordered_ops) {
    if (ordered_ops.size() != 2) {
        return false;
    }
    const size_t pointwise = ordered_ops[0];
    const size_t matmul = ordered_ops[1];
    if (!IsUnaryPointwiseOp(problem, pointwise) || !IsMatMul(problem.ops[matmul])) {
        return false;
    }
    const size_t output_tensor = problem.ops[pointwise].outputs.front();
    if (output_tensor >= graph.consumers_of_tensor.size()) {
        return false;
    }
    const auto& consumers = graph.consumers_of_tensor[output_tensor];
    if (consumers.size() != 1 || consumers.front() != matmul) {
        return false;
    }
    if (problem.ops[matmul].inputs.size() < 2 ||
        problem.ops[matmul].inputs.front() != output_tensor) {
        return false;
    }
    const auto& pointwise_out = problem.tensors[output_tensor];
    const auto& matmul_activation = problem.tensors[problem.ops[matmul].inputs.front()];
    if (pointwise_out.width != matmul_activation.width ||
        pointwise_out.height != matmul_activation.height) {
        return false;
    }
    return true;
}

size_t GetEvaluationOutputOpId(const Problem& problem, const OpGraph& graph,
                               const std::vector<size_t>& ordered_ops) {
    if (IsPreOpFusionChain(problem, graph, ordered_ops)) {
        return ordered_ops.back();
    }
    return ordered_ops.front();
}

int64_t MatMulK(const Problem& problem, size_t op_id) {
    const auto& op = problem.ops[op_id];
    if (op.inputs.size() < 2) {
        return 1;
    }
    return std::max<int64_t>(1, problem.tensors[op.inputs[0]].width);
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

std::vector<size_t> TopoSortSubset(const OpGraph& graph,
                                   const std::vector<size_t>& ops) {
    std::vector<size_t> ordered = ops;
    std::sort(ordered.begin(), ordered.end(),
              [&](size_t lhs, size_t rhs) {
                  return graph.topo_pos[lhs] < graph.topo_pos[rhs];
              });
    return ordered;
}

bool SharesCommonOutputShape(const Problem& problem,
                             const std::vector<size_t>& ordered_ops) {
    if (ordered_ops.empty()) {
        return false;
    }
    const auto& ref_tensor =
        problem.tensors[problem.ops[ordered_ops.front()].outputs.front()];
    for (size_t op_id : ordered_ops) {
        if (problem.ops[op_id].outputs.size() != 1) {
            return false;
        }
        const auto& tensor = problem.tensors[problem.ops[op_id].outputs.front()];
        if (tensor.width != ref_tensor.width || tensor.height != ref_tensor.height) {
            return false;
        }
    }
    return true;
}

bool IsConnectedSubgraph(const OpGraph& graph, const std::vector<size_t>& ops) {
    if (ops.empty()) {
        return false;
    }
    if (ops.size() == 1) {
        return true;
    }

    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    std::unordered_set<size_t> visited;
    std::queue<size_t> q;
    q.push(ops.front());
    visited.insert(ops.front());

    while (!q.empty()) {
        size_t op_id = q.front();
        q.pop();
        for (size_t succ : graph.succs[op_id]) {
            if (op_set.count(succ) && !visited.count(succ)) {
                visited.insert(succ);
                q.push(succ);
            }
        }
        for (size_t pred : graph.preds[op_id]) {
            if (op_set.count(pred) && !visited.count(pred)) {
                visited.insert(pred);
                q.push(pred);
            }
        }
    }

    return visited.size() == ops.size();
}

BoundaryInfo ComputeBoundaryInfo(const Problem& problem, const OpGraph& graph,
                                 const std::vector<size_t>& ordered_ops) {
    BoundaryInfo info;
    std::unordered_set<size_t> op_set(ordered_ops.begin(), ordered_ops.end());
    std::unordered_set<size_t> boundary_inputs;
    std::unordered_set<size_t> boundary_outputs;
    std::unordered_set<size_t> internal_tensors;

    for (size_t op_id : ordered_ops) {
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

    info.boundary_inputs.assign(boundary_inputs.begin(), boundary_inputs.end());
    info.boundary_outputs.assign(boundary_outputs.begin(), boundary_outputs.end());
    info.internal_tensors.assign(internal_tensors.begin(), internal_tensors.end());

    std::sort(info.boundary_inputs.begin(), info.boundary_inputs.end());
    std::sort(info.boundary_outputs.begin(), info.boundary_outputs.end());
    std::sort(info.internal_tensors.begin(), info.internal_tensors.end());

    return info;
}

bool ValidateTraversalOrder(size_t spatial_tiles,
                            const std::optional<TraversalOrder>& traversal) {
    if (!traversal.has_value()) {
        return true;
    }
    if (traversal->size() != spatial_tiles) {
        return false;
    }
    std::vector<int> seen(spatial_tiles, 0);
    for (int64_t idx : traversal.value()) {
        if (idx < 0 || idx >= static_cast<int64_t>(spatial_tiles)) {
            return false;
        }
        if (seen[idx]++) {
            return false;
        }
    }
    return true;
}

std::vector<size_t> BuildTraversal(size_t spatial_tiles,
                                   const std::optional<TraversalOrder>& traversal) {
    std::vector<size_t> order;
    order.reserve(spatial_tiles);
    if (!traversal.has_value()) {
        for (size_t i = 0; i < spatial_tiles; ++i) {
            order.push_back(i);
        }
        return order;
    }
    for (int64_t value : traversal.value()) {
        order.push_back(static_cast<size_t>(value));
    }
    return order;
}

bool CompareResidentUsage(
    const std::pair<SliceKey, ResidentSlice>& lhs,
    const std::pair<SliceKey, ResidentSlice>& rhs) {
    return lhs.second.last_used < rhs.second.last_used;
}

bool EnsureCapacity(
    int64_t capacity, int64_t base_bytes, int64_t output_active_bytes,
    const std::unordered_set<SliceKey, SliceKeyHash>& required_now,
    std::unordered_map<SliceKey, ResidentSlice, SliceKeyHash>* resident,
    int64_t incoming_new_bytes) {
    auto resident_total = [&]() {
        int64_t total = 0;
        for (const auto& [_, slice] : *resident) {
            total += slice.bytes;
        }
        return total;
    };

    while (base_bytes + output_active_bytes + resident_total() + incoming_new_bytes >
           capacity) {
        std::vector<std::pair<SliceKey, ResidentSlice>> evictable;
        for (const auto& entry : *resident) {
            if (!required_now.count(entry.first)) {
                evictable.push_back(entry);
            }
        }
        if (evictable.empty()) {
            return false;
        }
        auto victim = std::min_element(evictable.begin(), evictable.end(),
                                       CompareResidentUsage);
        resident->erase(victim->first);
    }
    return true;
}

StepRequirement BuildStepRequirement(
    const Problem& problem, const OpGraph& graph,
    const std::vector<size_t>& ordered_ops, const BoundaryInfo& boundary,
    const Granularity& granularity, const std::unordered_set<size_t>& retained_prev,
    const std::unordered_set<size_t>& retained_now, int64_t row_tile,
    int64_t col_tile, int64_t actual_h, int64_t actual_w, int64_t k_begin,
    int64_t k_step, int64_t step_index, int64_t max_k_steps) {
    StepRequirement requirement;
    std::unordered_set<size_t> op_set(ordered_ops.begin(), ordered_ops.end());
    std::unordered_set<size_t> boundary_outputs(boundary.boundary_outputs.begin(),
                                                boundary.boundary_outputs.end());

    const int64_t native_w = std::max<int64_t>(1, problem.native_granularity.width);
    const int64_t native_h = std::max<int64_t>(1, problem.native_granularity.height);
    const int64_t native_k = std::max<int64_t>(1, problem.native_granularity.depth);
    const double spatial_factor =
        static_cast<double>(CeilDiv(granularity.width, native_w) *
                            CeilDiv(granularity.height, native_h));

    for (size_t op_id : ordered_ops) {
        const auto& op = problem.ops[op_id];
        const bool final_k_step = (step_index == max_k_steps - 1);

        if (IsPointwise(op)) {
            if (!final_k_step) {
                continue;
            }
            requirement.compute_cost += static_cast<double>(op.base_cost) * spatial_factor;
            for (size_t tensor_id : op.inputs) {
                int producer = graph.producer_of_tensor[tensor_id];
                if (producer >= 0 && op_set.count(static_cast<size_t>(producer))) {
                    continue;
                }
                if (retained_prev.count(tensor_id)) {
                    continue;
                }
                requirement.slices[{tensor_id, 0, row_tile, col_tile, 0}] =
                    actual_h * actual_w;
            }
            size_t output_tensor = op.outputs.front();
            if (boundary_outputs.count(output_tensor)) {
                requirement.output_active_bytes += actual_h * actual_w;
                if (retained_now.count(output_tensor)) {
                    requirement.retained_outputs_completed.push_back(output_tensor);
                } else {
                    requirement.output_write_bytes += actual_h * actual_w;
                }
            }
        } else if (IsMatMul(op)) {
            requirement.compute_cost += static_cast<double>(op.base_cost) * spatial_factor *
                                        (static_cast<double>(k_step) /
                                         static_cast<double>(native_k));

            for (size_t input_index = 0; input_index < op.inputs.size(); ++input_index) {
                size_t tensor_id = op.inputs[input_index];
                int producer = graph.producer_of_tensor[tensor_id];
                if (producer >= 0 && op_set.count(static_cast<size_t>(producer))) {
                    continue;
                }
                if (retained_prev.count(tensor_id)) {
                    continue;
                }

                if (input_index == 0) {
                    requirement.slices[{tensor_id, 1, row_tile, 0, k_begin}] =
                        actual_h * k_step;
                } else {
                    requirement.slices[{tensor_id, 2, 0, col_tile, k_begin}] =
                        actual_w * k_step;
                }
            }

            size_t output_tensor = op.outputs.front();
            if (boundary_outputs.count(output_tensor)) {
                requirement.output_active_bytes += actual_h * actual_w;
                if (final_k_step) {
                    if (retained_now.count(output_tensor)) {
                        requirement.retained_outputs_completed.push_back(output_tensor);
                    } else {
                        requirement.output_write_bytes += actual_h * actual_w;
                    }
                }
            }
        } else {
            return {};
        }
    }

    return requirement;
}

bool Contains(const std::unordered_set<size_t>& values, size_t value) {
    return values.find(value) != values.end();
}

bool EvaluateSubgraphImpl(const Problem& problem, const OpGraph& graph,
                          const Subgraph& sg,
                          const std::vector<size_t>& retained_inputs_vec,
                          SubgraphEvaluation* evaluation) {
    if (evaluation == nullptr) {
        return false;
    }
    *evaluation = {};

    if (sg.op_ids.empty()) {
        return false;
    }
    if (sg.granularity.width <= 0 || sg.granularity.height <= 0 ||
        sg.granularity.depth <= 0) {
        return false;
    }

    std::unordered_set<size_t> unique_ops;
    for (size_t op_id : sg.op_ids) {
        if (op_id >= problem.ops.size()) {
            return false;
        }
        if (!unique_ops.insert(op_id).second) {
            return false;
        }
    }

    std::vector<size_t> ordered_ops = TopoSortSubset(graph, sg.op_ids);
    if (!IsConnectedSubgraph(graph, ordered_ops)) {
        return false;
    }
    const bool allow_preop_fusion =
        ReadBoolEnvOrDefault("MLSYS_ENABLE_PREOP_FUSION", true);
    if (!SharesCommonOutputShape(problem, ordered_ops) &&
        !(allow_preop_fusion && IsPreOpFusionChain(problem, graph, ordered_ops))) {
        return false;
    }

    BoundaryInfo boundary = ComputeBoundaryInfo(problem, graph, ordered_ops);
    std::unordered_set<size_t> boundary_outputs_set(boundary.boundary_outputs.begin(),
                                                    boundary.boundary_outputs.end());
    std::unordered_set<size_t> retained_now;
    for (size_t tensor_id : sg.tensors_to_retain) {
        if (!boundary_outputs_set.count(tensor_id)) {
            return false;
        }
        retained_now.insert(tensor_id);
    }

    std::unordered_set<size_t> retained_prev;
    for (size_t tensor_id : retained_inputs_vec) {
        retained_prev.insert(tensor_id);
    }

    const size_t output_op_id = GetEvaluationOutputOpId(problem, graph, ordered_ops);
    const auto& ref_out = problem.tensors[problem.ops[output_op_id].outputs.front()];
    const int64_t tiles_w = CeilDiv(ref_out.width, sg.granularity.width);
    const int64_t tiles_h = CeilDiv(ref_out.height, sg.granularity.height);
    const size_t spatial_tiles = static_cast<size_t>(tiles_w * tiles_h);
    if (!ValidateTraversalOrder(spatial_tiles, sg.traversal_order)) {
        return false;
    }
    const auto traversal = BuildTraversal(spatial_tiles, sg.traversal_order);

    std::unordered_set<size_t> retained_prev_used;
    int64_t retained_prev_bytes = 0;
    for (size_t tensor_id : boundary.boundary_inputs) {
        if (Contains(retained_prev, tensor_id)) {
            retained_prev_used.insert(tensor_id);
            retained_prev_bytes += TensorBytes(problem.tensors[tensor_id]);
        }
    }

    int64_t max_k_steps = 1;
    for (size_t op_id : ordered_ops) {
        if (IsMatMul(problem.ops[op_id])) {
            max_k_steps = std::max(max_k_steps,
                                   CeilDiv(MatMulK(problem, op_id),
                                           sg.granularity.depth));
        }
    }

    std::unordered_map<SliceKey, ResidentSlice, SliceKeyHash> resident;
    std::unordered_map<size_t, int64_t> retained_output_bytes;
    TotalLatency subgraph_latency = 0.0;
    int64_t logical_time = 0;
    int64_t peak_live_bytes = retained_prev_bytes;

    auto resident_total = [&resident]() {
        int64_t total = 0;
        for (const auto& [_, slice] : resident) {
            total += slice.bytes;
        }
        return total;
    };

    for (size_t traversal_idx : traversal) {
        const int64_t row_tile = static_cast<int64_t>(traversal_idx / tiles_w);
        const int64_t col_tile = static_cast<int64_t>(traversal_idx % tiles_w);
        const int64_t actual_h =
            std::min<int64_t>(sg.granularity.height,
                              ref_out.height - row_tile * sg.granularity.height);
        const int64_t actual_w =
            std::min<int64_t>(sg.granularity.width,
                              ref_out.width - col_tile * sg.granularity.width);

        for (int64_t step_index = 0; step_index < max_k_steps; ++step_index) {
            const int64_t k_begin = step_index * sg.granularity.depth;
            int64_t actual_k_step = 0;
            for (size_t op_id : ordered_ops) {
                if (!IsMatMul(problem.ops[op_id])) {
                    continue;
                }
                const int64_t op_k = MatMulK(problem, op_id);
                if (k_begin < op_k) {
                    actual_k_step = std::max<int64_t>(
                        actual_k_step,
                        std::min<int64_t>(sg.granularity.depth, op_k - k_begin));
                }
            }
            if (actual_k_step == 0) {
                actual_k_step = sg.granularity.depth;
            }

            StepRequirement requirement = BuildStepRequirement(
                problem, graph, ordered_ops, boundary, sg.granularity,
                retained_prev_used, retained_now, row_tile, col_tile, actual_h,
                actual_w, k_begin, actual_k_step, step_index, max_k_steps);

            std::unordered_set<SliceKey, SliceKeyHash> required_now;
            int64_t new_bytes = 0;
            for (const auto& [key, bytes] : requirement.slices) {
                required_now.insert(key);
                if (!resident.count(key)) {
                    new_bytes += bytes;
                }
            }

            int64_t retained_output_total = 0;
            for (const auto& [_, bytes] : retained_output_bytes) {
                retained_output_total += bytes;
            }

            if (!EnsureCapacity(problem.fast_memory_capacity,
                                retained_prev_bytes + retained_output_total,
                                requirement.output_active_bytes, required_now,
                                &resident, new_bytes)) {
                return false;
            }

            double memory_in = 0.0;
            for (const auto& [key, bytes] : requirement.slices) {
                auto it = resident.find(key);
                if (it == resident.end()) {
                    resident[key] = ResidentSlice{bytes, ++logical_time};
                    memory_in += static_cast<double>(bytes);
                } else {
                    it->second.last_used = ++logical_time;
                }
            }

            peak_live_bytes = std::max(
                peak_live_bytes,
                retained_prev_bytes + retained_output_total +
                    requirement.output_active_bytes + resident_total());

            if (retained_prev_bytes + retained_output_total +
                    requirement.output_active_bytes >
                problem.fast_memory_capacity) {
                return false;
            }

            const double step_latency = std::max(
                requirement.compute_cost,
                (memory_in + static_cast<double>(requirement.output_write_bytes)) /
                    static_cast<double>(problem.slow_memory_bandwidth));
            subgraph_latency += step_latency;

            for (size_t tensor_id : requirement.retained_outputs_completed) {
                retained_output_bytes[tensor_id] += actual_h * actual_w;
            }
        }
    }

    for (size_t tensor_id : sg.tensors_to_retain) {
        if (retained_output_bytes[tensor_id] != TensorBytes(problem.tensors[tensor_id])) {
            return false;
        }
    }

    evaluation->valid = true;
    evaluation->latency = subgraph_latency;
    evaluation->peak_live_bytes = peak_live_bytes;
    return true;
}

bool RecomputeImpl(const Problem& problem, const Solution& input_solution,
                   Solution* normalized_solution, TotalLatency* total_latency) {
    if (input_solution.subgraphs.empty()) {
        std::cerr << "ERROR: Solution contains no subgraphs\n";
        return false;
    }

    const OpGraph graph = BuildOpGraph(problem);
    std::set<size_t> covered_ops;
    std::unordered_set<size_t> slow_available(problem.graph_inputs.begin(),
                                              problem.graph_inputs.end());
    std::unordered_set<size_t> retained_prev;
    TotalLatency recomputed_total = 0.0;

    if (normalized_solution != nullptr) {
        *normalized_solution = input_solution;
    }

    for (size_t sg_idx = 0; sg_idx < input_solution.subgraphs.size(); ++sg_idx) {
        const auto& sg = input_solution.subgraphs[sg_idx];
        if (sg.op_ids.empty()) {
            std::cerr << "ERROR: Subgraph " << sg_idx << " is empty\n";
            return false;
        }
        if (sg.granularity.width <= 0 || sg.granularity.height <= 0 ||
            sg.granularity.depth <= 0) {
            std::cerr << "ERROR: Subgraph " << sg_idx
                      << " has non-positive granularity\n";
            return false;
        }

        std::unordered_set<size_t> unique_ops;
        for (size_t op_id : sg.op_ids) {
            if (op_id >= problem.ops.size()) {
                std::cerr << "ERROR: Subgraph " << sg_idx << " references invalid op "
                          << op_id << "\n";
                return false;
            }
            if (!unique_ops.insert(op_id).second) {
                std::cerr << "ERROR: Subgraph " << sg_idx
                          << " repeats an op within the same group\n";
                return false;
            }
            covered_ops.insert(op_id);
        }

        std::vector<size_t> ordered_ops = TopoSortSubset(graph, sg.op_ids);
        if (!IsConnectedSubgraph(graph, ordered_ops)) {
            std::cerr << "ERROR: Subgraph " << sg_idx
                      << " is not a connected sub-DAG\n";
            return false;
        }
        const bool allow_preop_fusion =
            ReadBoolEnvOrDefault("MLSYS_ENABLE_PREOP_FUSION", true);
        if (!SharesCommonOutputShape(problem, ordered_ops) &&
            !(allow_preop_fusion && IsPreOpFusionChain(problem, graph, ordered_ops))) {
            std::cerr << "ERROR: Subgraph " << sg_idx
                      << " mixes incompatible output shapes\n";
            return false;
        }

        BoundaryInfo boundary = ComputeBoundaryInfo(problem, graph, ordered_ops);
        std::unordered_set<size_t> boundary_outputs_set(boundary.boundary_outputs.begin(),
                                                        boundary.boundary_outputs.end());
        std::unordered_set<size_t> retained_now;
        for (size_t tensor_id : sg.tensors_to_retain) {
            if (!boundary_outputs_set.count(tensor_id)) {
                std::cerr << "ERROR: Subgraph " << sg_idx
                          << " retains tensor " << tensor_id
                          << " that is not one of its boundary outputs\n";
                return false;
            }
            retained_now.insert(tensor_id);
        }

        for (size_t tensor_id : boundary.boundary_inputs) {
            if (!Contains(retained_prev, tensor_id) &&
                !Contains(slow_available, tensor_id)) {
                std::cerr << "ERROR: Subgraph " << sg_idx
                          << " needs tensor " << tensor_id
                          << " that is unavailable in fast or slow memory\n";
                return false;
            }
        }

        std::vector<size_t> retained_inputs_for_sg;
        for (size_t tensor_id : boundary.boundary_inputs) {
            if (Contains(retained_prev, tensor_id)) {
                retained_inputs_for_sg.push_back(tensor_id);
            }
        }

        SubgraphEvaluation subgraph_eval;
        if (!EvaluateSubgraphImpl(problem, graph, sg, retained_inputs_for_sg,
                                  &subgraph_eval) ||
            !subgraph_eval.valid) {
            std::cerr << "ERROR: Subgraph " << sg_idx
                      << " exceeds fast memory capacity\n";
            return false;
        }
        const double subgraph_latency = subgraph_eval.latency;

        if (std::fabs(subgraph_latency - sg.subgraph_latency) > 1e-3) {
            std::cerr << "WARNING: Subgraph " << sg_idx
                      << " reported latency " << sg.subgraph_latency
                      << " but recomputed latency is " << subgraph_latency << "\n";
        }
        if (normalized_solution != nullptr) {
            (*normalized_solution).subgraphs[sg_idx].subgraph_latency =
                subgraph_latency;
        }

        for (size_t tensor_id : boundary.boundary_outputs) {
            if (!retained_now.count(tensor_id)) {
                slow_available.insert(tensor_id);
            }
        }
        retained_prev = std::move(retained_now);
        recomputed_total += subgraph_latency;
    }

    for (size_t op_id = 0; op_id < problem.ops.size(); ++op_id) {
        if (!covered_ops.count(op_id)) {
            std::cerr << "ERROR: Op " << op_id << " is not covered by any subgraph\n";
            return false;
        }
    }

    for (size_t tensor_id : problem.graph_outputs) {
        if (!slow_available.count(tensor_id)) {
            std::cerr << "ERROR: Graph output tensor " << tensor_id
                      << " is not available in slow memory at program end\n";
            return false;
        }
    }

    if (total_latency != nullptr) {
        *total_latency = recomputed_total;
    }
    return true;
}

}  // namespace

TotalLatency Evaluate(const Problem& problem, const Solution& solution) {
    TotalLatency total_latency = -1.0;
    if (!RecomputeImpl(problem, solution, nullptr, &total_latency)) {
        return -1.0;
    }
    return total_latency;
}

bool EvaluateSubgraph(const Problem& problem, const Subgraph& subgraph,
                      const std::vector<size_t>& retained_inputs,
                      SubgraphEvaluation* evaluation) {
    const OpGraph graph = BuildOpGraph(problem);
    return EvaluateSubgraphImpl(problem, graph, subgraph, retained_inputs,
                                evaluation);
}

bool RecomputeLatencies(const Problem& problem, Solution* solution,
                        TotalLatency* total_latency) {
    if (solution == nullptr) {
        return false;
    }
    Solution normalized;
    if (!RecomputeImpl(problem, *solution, &normalized, total_latency)) {
        return false;
    }
    *solution = std::move(normalized);
    return true;
}

}  // namespace mlsys
