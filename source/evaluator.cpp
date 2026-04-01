#include "evaluator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlsys {

namespace {

int64_t CeilDiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t TileWidth(Width tensor_w, Width gran_w, size_t col_idx) {
    const int64_t start = static_cast<int64_t>(col_idx) * gran_w;
    return std::min<int64_t>(gran_w, tensor_w - start);
}

int64_t TileHeight(Height tensor_h, Height gran_h, size_t row_idx) {
    const int64_t start = static_cast<int64_t>(row_idx) * gran_h;
    return std::min<int64_t>(gran_h, tensor_h - start);
}

std::vector<size_t> TopologicalOrder(const Problem& problem,
                                     const std::unordered_set<size_t>& op_set) {
    std::unordered_map<size_t, std::vector<size_t>> tensor_to_consumers;
    for (size_t op_id : op_set) {
        for (size_t input_tensor : problem.ops[op_id].inputs) {
            tensor_to_consumers[input_tensor].push_back(op_id);
        }
    }

    std::unordered_map<size_t, int> in_degree;
    for (size_t op_id : op_set) {
        in_degree[op_id] = 0;
    }
    for (size_t op_id : op_set) {
        for (size_t out_tensor : problem.ops[op_id].outputs) {
            auto it = tensor_to_consumers.find(out_tensor);
            if (it == tensor_to_consumers.end()) continue;
            for (size_t consumer : it->second) {
                if (consumer != op_id) {
                    ++in_degree[consumer];
                }
            }
        }
    }

    std::queue<size_t> q;
    for (auto& [op_id, deg] : in_degree) {
        if (deg == 0) q.push(op_id);
    }

    std::vector<size_t> order;
    order.reserve(op_set.size());
    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        order.push_back(u);

        for (size_t out_tensor : problem.ops[u].outputs) {
            auto it = tensor_to_consumers.find(out_tensor);
            if (it == tensor_to_consumers.end()) continue;
            for (size_t v : it->second) {
                if (v == u) continue;
                if (--in_degree[v] == 0) {
                    q.push(v);
                }
            }
        }
    }

    return order;
}

struct SliceInfo {
    int64_t transfer_bytes = 0;
    int64_t resident_bytes = 0;
    bool pinned = false;
    int64_t last_used = 0;
};

std::string MakeKey(size_t tensor_id, const std::string& role, size_t a, size_t b) {
    return std::to_string(tensor_id) + ":" + role + ":" + std::to_string(a) + ":" +
           std::to_string(b);
}

std::string MakeKey(size_t tensor_id, const std::string& role, size_t a) {
    return std::to_string(tensor_id) + ":" + role + ":" + std::to_string(a);
}

struct TensorRoleMask {
    bool pointwise = false;
    bool lhs = false;
    bool rhs = false;
};

std::vector<size_t> TileTraversal(size_t rows, size_t cols,
                                  const std::optional<TraversalOrder>& traversal_order) {
    std::vector<size_t> order;
    const size_t total = rows * cols;
    order.reserve(total);

    if (traversal_order.has_value() && !traversal_order->empty()) {
        for (int64_t idx : traversal_order.value()) {
            if (idx >= 0 && static_cast<size_t>(idx) < total) {
                order.push_back(static_cast<size_t>(idx));
            }
        }
    }

    if (order.size() != total) {
        order.clear();
        for (size_t i = 0; i < total; ++i) order.push_back(i);
    }

    return order;
}

double EstimateSubgraphLatency(const Problem& problem, const Solution& solution,
                               const Subgraph& sg,
                               const std::unordered_set<size_t>& retained_from_prev,
                               std::unordered_set<size_t>& retained_after) {
    (void)retained_from_prev;
    (void)solution;

    if (sg.op_ids.empty()) {
        retained_after.clear();
        return 0.0;
    }

    std::unordered_set<size_t> op_set(sg.op_ids.begin(), sg.op_ids.end());
    auto topo = TopologicalOrder(problem, op_set);
    if (topo.size() != op_set.size()) {
        std::cerr << "WARNING: subgraph is not a DAG or not connected properly\n";
        return -1.0;
    }

    // Determine tensors produced inside the subgraph and which of them are external outputs.
    std::unordered_set<size_t> produced_inside;
    std::unordered_set<size_t> consumed_inside;
    for (size_t op_id : op_set) {
        for (size_t t : problem.ops[op_id].outputs) produced_inside.insert(t);
        for (size_t t : problem.ops[op_id].inputs) consumed_inside.insert(t);
    }

    std::unordered_set<size_t> external_inputs;
    std::unordered_set<size_t> external_outputs;
    std::unordered_map<size_t, TensorRoleMask> input_roles;

    for (size_t op_id : topo) {
        const auto& op = problem.ops[op_id];
        for (size_t input_idx = 0; input_idx < op.inputs.size(); ++input_idx) {
            size_t t = op.inputs[input_idx];
            if (!produced_inside.count(t)) {
                external_inputs.insert(t);
                auto& mask = input_roles[t];
                if (op.op_type == "MatMul") {
                    if (input_idx == 0) mask.lhs = true;
                    if (input_idx == 1) mask.rhs = true;
                } else {
                    mask.pointwise = true;
                }
            }
        }
        for (size_t out_t : op.outputs) {
            bool consumed_outside = false;
            for (size_t consumer_id = 0; consumer_id < problem.ops.size(); ++consumer_id) {
                if (op_set.count(consumer_id)) continue;
                const auto& consumer = problem.ops[consumer_id];
                if (std::find(consumer.inputs.begin(), consumer.inputs.end(), out_t) !=
                    consumer.inputs.end()) {
                    consumed_outside = true;
                    break;
                }
            }
            if (consumed_outside || std::find(problem.graph_outputs.begin(), problem.graph_outputs.end(), out_t) != problem.graph_outputs.end()) {
                external_outputs.insert(out_t);
            }
        }
    }

    const auto& first_op = problem.ops[topo.front()];
    const Tensor& first_out_tensor = problem.tensors[first_op.outputs.front()];
    const size_t rows = static_cast<size_t>(CeilDiv(first_out_tensor.height, sg.granularity.height));
    const size_t cols = static_cast<size_t>(CeilDiv(first_out_tensor.width, sg.granularity.width));
    const auto tile_order = TileTraversal(rows, cols, sg.traversal_order);

    struct CacheEntry {
        SliceInfo info;
    };

    std::unordered_map<std::string, CacheEntry> cache;
    int64_t resident_bytes = 0;

    auto ensure_capacity = [&](int64_t needed, const std::unordered_set<std::string>& keep) -> bool {
        while (resident_bytes + needed > problem.fast_memory_capacity) {
            std::string victim;
            int64_t oldest = std::numeric_limits<int64_t>::max();
            for (const auto& [key, entry] : cache) {
                if (keep.count(key) || entry.info.pinned) continue;
                if (entry.info.last_used < oldest) {
                    oldest = entry.info.last_used;
                    victim = key;
                }
            }
            if (victim.empty()) return false;
            resident_bytes -= cache[victim].info.resident_bytes;
            cache.erase(victim);
        }
        return true;
    };

    double total = 0.0;
    int64_t access_clock = 0;

    for (size_t tile_idx : tile_order) {
        const size_t row = tile_idx / cols;
        const size_t col = tile_idx % cols;
        const int64_t tile_w = TileWidth(first_out_tensor.width, sg.granularity.width, col);
        const int64_t tile_h = TileHeight(first_out_tensor.height, sg.granularity.height, row);

        std::unordered_set<std::string> keep_this_tile;
        int64_t memory_cost = 0;

        // Boundary inputs: load slices once per tile if not already resident.
        for (size_t t : external_inputs) {
            const auto& roles = input_roles[t];
            const auto& tensor = problem.tensors[t];
            const int64_t k_total = tensor.width;  // width = columns, height = rows.
            if (roles.lhs) {
                std::string key = MakeKey(t, "lhs", row);
                int64_t transfer_bytes = tile_h * k_total;
                int64_t resident = tile_h * std::min<int64_t>(sg.granularity.depth, k_total);
                if (!cache.count(key)) {
                    if (!ensure_capacity(resident, keep_this_tile)) return -1.0;
                    cache[key] = CacheEntry{{transfer_bytes, resident, false, ++access_clock}};
                    resident_bytes += resident;
                    memory_cost += static_cast<int64_t>(std::ceil(static_cast<double>(transfer_bytes) /
                                                                 problem.slow_memory_bandwidth));
                } else {
                    cache[key].info.last_used = ++access_clock;
                }
                keep_this_tile.insert(key);
            }
            if (roles.rhs) {
                std::string key = MakeKey(t, "rhs", col);
                int64_t transfer_bytes = k_total * tile_w;
                int64_t resident = std::min<int64_t>(sg.granularity.depth, k_total) * tile_w;
                if (!cache.count(key)) {
                    if (!ensure_capacity(resident, keep_this_tile)) return -1.0;
                    cache[key] = CacheEntry{{transfer_bytes, resident, false, ++access_clock}};
                    resident_bytes += resident;
                    memory_cost += static_cast<int64_t>(std::ceil(static_cast<double>(transfer_bytes) /
                                                                 problem.slow_memory_bandwidth));
                } else {
                    cache[key].info.last_used = ++access_clock;
                }
                keep_this_tile.insert(key);
            }
            if (roles.pointwise) {
                std::string key = MakeKey(t, "pw", row, col);
                int64_t transfer_bytes = tile_w * tile_h;
                int64_t resident = transfer_bytes;
                if (!cache.count(key)) {
                    if (!ensure_capacity(resident, keep_this_tile)) return -1.0;
                    cache[key] = CacheEntry{{transfer_bytes, resident, false, ++access_clock}};
                    resident_bytes += resident;
                    memory_cost += static_cast<int64_t>(std::ceil(static_cast<double>(transfer_bytes) /
                                                                 problem.slow_memory_bandwidth));
                } else {
                    cache[key].info.last_used = ++access_clock;
                }
                keep_this_tile.insert(key);
            }
        }

        // External outputs: write slices unless retained.
        for (size_t out_t : external_outputs) {
            bool retain = std::find(sg.tensors_to_retain.begin(), sg.tensors_to_retain.end(), out_t) !=
                          sg.tensors_to_retain.end();
            if (retain) {
                retained_after.insert(out_t);
                continue;
            }
            std::string key = MakeKey(out_t, "out", row, col);
            int64_t transfer_bytes = tile_w * tile_h;
            int64_t resident = transfer_bytes;
            if (!cache.count(key)) {
                if (!ensure_capacity(resident, keep_this_tile)) return -1.0;
                cache[key] = CacheEntry{{transfer_bytes, resident, false, ++access_clock}};
                resident_bytes += resident;
            } else {
                cache[key].info.last_used = ++access_clock;
            }
            memory_cost += static_cast<int64_t>(std::ceil(static_cast<double>(transfer_bytes) /
                                                         problem.slow_memory_bandwidth));
            keep_this_tile.insert(key);
        }

        // Compute cost for this tile.
        double compute_cost = 0.0;
        for (size_t op_id : topo) {
            const auto& op = problem.ops[op_id];
            const Tensor& out_tensor = problem.tensors[op.outputs.front()];
            if (op.op_type == "MatMul") {
                const auto& lhs = problem.tensors[op.inputs[0]];
                const int64_t k_total = lhs.width;
                compute_cost += static_cast<double>(op.base_cost) *
                                static_cast<double>(k_total) /
                                static_cast<double>(problem.native_granularity.depth);
            } else {
                (void)out_tensor;
                compute_cost += static_cast<double>(op.base_cost);
            }
        }

        // Simple working-set validation for this tile.
        int64_t working_set = 0;
        for (size_t t : external_inputs) {
            const auto& roles = input_roles[t];
            const auto& tensor = problem.tensors[t];
            const int64_t k_total = tensor.width;
            if (roles.lhs) working_set += tile_h * std::min<int64_t>(sg.granularity.depth, k_total);
            if (roles.rhs) working_set += std::min<int64_t>(sg.granularity.depth, k_total) * tile_w;
            if (roles.pointwise) working_set += tile_w * tile_h;
        }
        working_set += tile_w * tile_h;
        if (working_set > problem.fast_memory_capacity) {
            std::cerr << "WARNING: subgraph tile working set exceeds fast memory capacity\n";
            return -1.0;
        }

        total += std::max(compute_cost, static_cast<double>(memory_cost));

        // Evict non-retained output slices immediately after the tile.
        std::vector<std::string> to_erase;
        for (const auto& [key, entry] : cache) {
            if (keep_this_tile.count(key)) continue;
            // Keep boundary inputs resident for reuse; evict only if they are not required.
            if (key.find(":out:") != std::string::npos) {
                resident_bytes -= entry.info.resident_bytes;
                to_erase.push_back(key);
            }
        }
        for (const auto& key : to_erase) cache.erase(key);
    }
    return total;
}

}  // namespace

std::vector<SubgraphLatency> ComputeSubgraphLatencies(const Problem& problem,
                                                      const Solution& solution) {
    std::vector<SubgraphLatency> latencies;
    latencies.reserve(solution.subgraphs.size());

    std::unordered_set<size_t> retained_from_prev;
    for (const auto& sg : solution.subgraphs) {
        std::unordered_set<size_t> retained_after;
        double latency = EstimateSubgraphLatency(problem, solution, sg, retained_from_prev,
                                                 retained_after);
        latencies.push_back(latency);
        retained_from_prev = std::move(retained_after);
    }

    return latencies;
}

TotalLatency Evaluate(const Problem& problem, const Solution& solution) {
    std::set<size_t> covered_ops;
    for (const auto& sg : solution.subgraphs) {
        for (size_t op_id : sg.op_ids) {
            covered_ops.insert(op_id);
        }
    }

    for (size_t i = 0; i < problem.ops.size(); ++i) {
        if (!covered_ops.count(i)) {
            std::cerr << "WARNING: Op " << i << " not covered by any subgraph\n";
            return -1.0;
        }
    }

    auto latencies = ComputeSubgraphLatencies(problem, solution);
    TotalLatency total = 0.0;
    for (double latency : latencies) {
        if (latency < 0.0) return -1.0;
        total += latency;
    }
    return total;
}

}  // namespace mlsys
