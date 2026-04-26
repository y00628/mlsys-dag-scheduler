#include "solver.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "optimus.h"

namespace mlsys {

namespace {

enum class SolverBackend {
    kBaseline = 0,
    kOptimus,
    kOptimusConv,
    kOptimusConvV2,
    kOptimusPaper,
    kOptimusSched,
};

struct BaselineCandidate {
    Granularity granularity{};
    double latency = std::numeric_limits<double>::infinity();
    bool valid = false;
};

int64_t CeilDivInt(int64_t value, int64_t divisor) {
    if (divisor <= 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

bool IsMatMul(const Op& op) {
    return op.op_type == "MatMul";
}

int64_t MatMulK(const Problem& problem, size_t op_id) {
    const auto& op = problem.ops[op_id];
    if (op.inputs.size() < 2) {
        return 1;
    }
    return std::max<int64_t>(1, problem.tensors[op.inputs[0]].width);
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

std::vector<int64_t> BuildKCandidates(const Problem& problem, size_t op_id) {
    if (!IsMatMul(problem.ops[op_id])) {
        return {1};
    }

    const int64_t max_k = MatMulK(problem, op_id);
    const int64_t native_k = std::max<int64_t>(1, problem.native_granularity.depth);
    std::set<int64_t> values;
    values.insert(max_k);
    values.insert(std::max<int64_t>(1, std::min(max_k, native_k)));
    values.insert(std::max<int64_t>(1, std::min(max_k, native_k / 2)));
    values.insert(std::max<int64_t>(1, std::min(max_k, native_k / 4)));

    std::vector<int64_t> result(values.begin(), values.end());
    std::sort(result.begin(), result.end(), std::greater<int64_t>());
    return result;
}

double EstimateSingleOpLatency(const Problem& problem, size_t op_idx,
                               const Granularity& granularity,
                               bool* valid_out) {
    const auto& op = problem.ops[op_idx];
    const auto& out = problem.tensors[op.outputs.front()];
    const int64_t native_w = std::max<int64_t>(1, problem.native_granularity.width);
    const int64_t native_h = std::max<int64_t>(1, problem.native_granularity.height);
    const int64_t native_k = std::max<int64_t>(1, problem.native_granularity.depth);

    const int64_t tiles_w = CeilDivInt(out.width, granularity.width);
    const int64_t tiles_h = CeilDivInt(out.height, granularity.height);
    const double spatial_factor =
        static_cast<double>(CeilDivInt(granularity.width, native_w) *
                            CeilDivInt(granularity.height, native_h));

    double total_latency = 0.0;
    bool valid = true;

    for (int64_t row_tile = 0; row_tile < tiles_h; ++row_tile) {
        for (int64_t col_tile = 0; col_tile < tiles_w; ++col_tile) {
            const int64_t actual_h =
                std::min<int64_t>(granularity.height,
                                  out.height - row_tile * granularity.height);
            const int64_t actual_w =
                std::min<int64_t>(granularity.width,
                                  out.width - col_tile * granularity.width);

            if (IsMatMul(op)) {
                const int64_t k_total = MatMulK(problem, op_idx);
                const int64_t k_steps = CeilDivInt(k_total, granularity.depth);
                for (int64_t step = 0; step < k_steps; ++step) {
                    const int64_t k_begin = step * granularity.depth;
                    const int64_t k_step =
                        std::min<int64_t>(granularity.depth, k_total - k_begin);
                    const int64_t working_set =
                        actual_h * k_step + actual_w * k_step + actual_h * actual_w;
                    if (working_set > problem.fast_memory_capacity) {
                        valid = false;
                        break;
                    }

                    const double compute =
                        static_cast<double>(op.base_cost) * spatial_factor *
                        (static_cast<double>(k_step) / static_cast<double>(native_k));
                    const double memory_in =
                        static_cast<double>(actual_h * k_step + actual_w * k_step);
                    const double memory_out =
                        (step == k_steps - 1) ? static_cast<double>(actual_h * actual_w)
                                              : 0.0;
                    total_latency += std::max(
                        compute, (memory_in + memory_out) /
                                     static_cast<double>(problem.slow_memory_bandwidth));
                }
                if (!valid) {
                    break;
                }
            } else {
                const int64_t working_set =
                    static_cast<int64_t>(op.inputs.size() + 1) * actual_h * actual_w;
                if (working_set > problem.fast_memory_capacity) {
                    valid = false;
                    break;
                }

                const double compute =
                    static_cast<double>(op.base_cost) * spatial_factor;
                const double memory_in =
                    static_cast<double>(op.inputs.size()) *
                    static_cast<double>(actual_h * actual_w);
                const double memory_out = static_cast<double>(actual_h * actual_w);
                total_latency += std::max(
                    compute, (memory_in + memory_out) /
                                 static_cast<double>(problem.slow_memory_bandwidth));
            }
        }
        if (!valid) {
            break;
        }
    }

    if (valid_out != nullptr) {
        *valid_out = valid;
    }
    return valid ? total_latency : std::numeric_limits<double>::infinity();
}

BaselineCandidate ChooseSingleOpGranularity(const Problem& problem, size_t op_idx) {
    BaselineCandidate best;
    const auto& out = problem.tensors[problem.ops[op_idx].outputs.front()];
    const auto widths =
        BuildSpatialCandidates(out.width, problem.native_granularity.width);
    const auto heights =
        BuildSpatialCandidates(out.height, problem.native_granularity.height);
    const auto depths = BuildKCandidates(problem, op_idx);

    for (int64_t width : widths) {
        for (int64_t height : heights) {
            for (int64_t depth : depths) {
                Granularity granularity{width, height, depth};
                bool valid = false;
                const double latency =
                    EstimateSingleOpLatency(problem, op_idx, granularity, &valid);
                if (!valid) {
                    continue;
                }
                if (!best.valid || latency < best.latency) {
                    best.valid = true;
                    best.granularity = granularity;
                    best.latency = latency;
                }
            }
        }
    }

    return best;
}

// -----------------------------------------------------------------
// Utility: topological sort of the ops DAG
// -----------------------------------------------------------------
std::vector<size_t> TopologicalSort(const Problem& problem) {
    size_t n = problem.ops.size();
    std::vector<int> in_degree(n, 0);

    // Build adjacency: op i -> op j if any output of i is an input of j
    std::unordered_map<size_t, std::vector<size_t>> tensor_to_consumer;
    for (size_t i = 0; i < n; ++i) {
        for (auto t : problem.ops[i].inputs) {
            tensor_to_consumer[t].push_back(i);
        }
    }

    std::vector<std::vector<size_t>> adj(n);
    for (size_t i = 0; i < n; ++i) {
        for (auto out_t : problem.ops[i].outputs) {
            for (auto consumer : tensor_to_consumer[out_t]) {
                adj[i].push_back(consumer);
                in_degree[consumer]++;
            }
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < n; ++i) {
        if (in_degree[i] == 0) q.push(i);
    }

    std::vector<size_t> order;
    order.reserve(n);
    while (!q.empty()) {
        auto u = q.front();
        q.pop();
        order.push_back(u);
        for (auto v : adj[u]) {
            if (--in_degree[v] == 0) q.push(v);
        }
    }

    return order;
}

// -----------------------------------------------------------------
// Naive baseline: each op gets its own subgraph with native granularity.
// This is always valid but likely far from optimal. Use as a starting
// point to verify I/O correctness, then improve.
// -----------------------------------------------------------------
Solution NaiveBaseline(const Problem& problem) {
    Solution sol;
    auto order = TopologicalSort(problem);

    for (auto op_idx : order) {
        Subgraph sg;
        sg.op_ids = {op_idx};
        const BaselineCandidate candidate =
            ChooseSingleOpGranularity(problem, op_idx);
        if (candidate.valid) {
            sg.granularity = candidate.granularity;
            sg.subgraph_latency = candidate.latency;
        } else {
            sg.granularity = problem.native_granularity;
            sg.granularity.depth = IsMatMul(problem.ops[op_idx])
                                       ? std::max<int64_t>(1, MatMulK(problem, op_idx))
                                       : 1;
            sg.subgraph_latency = std::numeric_limits<double>::infinity();
        }

        // Don't retain anything — every tensor goes back to slow memory
        sg.tensors_to_retain = {};

        sol.subgraphs.push_back(std::move(sg));
    }

    return sol;
}

SolverBackend ParseSolverBackend() {
    const char* raw = std::getenv("MLSYS_SOLVER");
    if (raw == nullptr) {
        return SolverBackend::kOptimusPaper;
    }

    std::string backend(raw);
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (backend == "baseline" || backend == "naive") {
        return SolverBackend::kBaseline;
    }
    if (backend == "optimus_conv" || backend == "optimus-conv" ||
        backend == "paper") {
        return SolverBackend::kOptimusConv;
    }
    if (backend == "optimus_conv_v2" || backend == "optimus-conv-v2" ||
        backend == "paper_v2") {
        return SolverBackend::kOptimusConvV2;
    }
    if (backend == "optimus_paper" || backend == "paper_optimus" ||
        backend == "paper_new") {
        return SolverBackend::kOptimusPaper;
    }
    if (backend == "optimus_sched" || backend == "optimus-sched" ||
        backend == "sched") {
        return SolverBackend::kOptimusSched;
    }
    return SolverBackend::kOptimus;
}

}  // namespace

// -----------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------
Solution Solve(const Problem& problem) {
    std::cerr << "Problem: " << problem.tensors.size() << " tensors, "
              << problem.ops.size() << " ops, "
              << "fast_mem=" << problem.fast_memory_capacity
              << ", slow_bw=" << problem.slow_memory_bandwidth << "\n";

    Solution best;
    switch (ParseSolverBackend()) {
        case SolverBackend::kBaseline:
            std::cerr << "Solver backend: baseline\n";
            best = NaiveBaseline(problem);
            break;
        case SolverBackend::kOptimus:
            std::cerr << "Solver backend: optimus\n";
            best = SolveWithOptimus(problem);
            break;
        case SolverBackend::kOptimusConv:
            std::cerr << "Solver backend: optimus_conv\n";
            best = SolveWithOptimusConvGuidance(problem);
            break;
        case SolverBackend::kOptimusConvV2:
            std::cerr << "Solver backend: optimus_conv_v2\n";
            best = SolveWithOptimusConvRerankV2(problem);
            break;
        case SolverBackend::kOptimusPaper:
            std::cerr << "Solver backend: optimus_paper\n";
            best = SolveWithPaperOptimus(problem);
            break;
        case SolverBackend::kOptimusSched:
            std::cerr << "Solver backend: optimus_sched\n";
            best = SolveWithOptimusSched(problem);
            break;
        default:
            std::cerr << "Solver backend: optimus\n";
            best = SolveWithOptimus(problem);
            break;
    }

    // Summary printed in main.cpp with ops coverage info.
    return best;
}

}  // namespace mlsys
