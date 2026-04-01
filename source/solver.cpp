#include "solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "evaluator.h"

namespace mlsys {

namespace {

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
        const auto& op = problem.ops[op_idx];
        Subgraph sg;
        sg.op_ids = {op_idx};
        sg.granularity.width = std::min<Width>(problem.native_granularity.width, 32);
        sg.granularity.height = std::min<Height>(problem.native_granularity.height, 32);

        // For MatMul, set depth = native width (full reduction in one step for now).
        // We keep a conservative depth so the current baseline fits more benchmarks.
        if (op.op_type == "MatMul" && op.inputs.size() == 2) {
            sg.granularity.depth = std::min<Depth>(problem.native_granularity.depth, 32);
        } else {
            sg.granularity.depth = 1;  // Pointwise: k is irrelevant
        }

        // Don't retain anything — every tensor goes back to slow memory
        sg.tensors_to_retain = {};

        sol.subgraphs.push_back(std::move(sg));
    }

    auto latencies = ComputeSubgraphLatencies(problem, sol);
    for (size_t i = 0; i < sol.subgraphs.size() && i < latencies.size(); ++i) {
        sol.subgraphs[i].subgraph_latency = latencies[i];
    }

    return sol;
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

    // Start with naive baseline
    Solution best = NaiveBaseline(problem);

    // TODO: Implement better strategies:
    //   1. Operator fusion — group connected ops into subgraphs to make
    //      intermediate tensors ephemeral (zero memory cost).
    //   2. Granularity tuning — find the best [w, h, k] per subgraph
    //      that fits the working set in fast memory.
    //   3. Tensor retention — keep frequently reused tensors in fast memory
    //      across subgraph boundaries.
    //   4. Traversal order optimization — snake patterns etc. for data reuse.
    //   5. Search / optimization — simulated annealing, beam search, ILP, etc.

    std::cerr << "Solution: " << best.subgraphs.size() << " subgraphs\n";
    return best;
}

}  // namespace mlsys
