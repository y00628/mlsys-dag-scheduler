#include "evaluator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <unordered_set>

namespace mlsys {

namespace {

// Compute the number of spatial tiles for a given tensor and granularity.
int64_t NumTilesW(Width tensor_w, Width gran_w) {
    return (tensor_w + gran_w - 1) / gran_w;
}

int64_t NumTilesH(Height tensor_h, Height gran_h) {
    return (tensor_h + gran_h - 1) / gran_h;
}

// Size of a single tile (in memory units = width * height).
int64_t TileSize(Width w, Height h) {
    return w * h;
}

}  // namespace

TotalLatency Evaluate(const Problem& problem, const Solution& solution) {
    // TODO: Implement the full evaluation logic matching the official scorer.
    //
    // High-level steps per subgraph:
    //   1. Determine boundary tensors (inputs from slow memory, outputs to slow memory)
    //      minus any tensors retained from previous subgraphs.
    //   2. Compute memory transfer cost = boundary_data_size / slow_memory_bandwidth.
    //   3. Compute execution cost based on base_costs, granularity, and tiling.
    //      - For MatMul: split-k steps = ceil(K / k), spatial tiles = ceil(W/w) * ceil(H/h)
    //      - For Pointwise: spatial tiles only, k ignored
    //      - Padding penalty if granularity < native_granularity
    //   4. Subgraph latency = max(compute_cost, memory_cost)  [roofline model]
    //   5. Total latency = sum of all subgraph latencies
    //
    // Also validate:
    //   - Every op is covered by at least one subgraph
    //   - Working set fits in fast_memory_capacity for each tile
    //   - Subgraph ops form a connected sub-DAG

    TotalLatency total = 0.0;

    std::set<size_t> covered_ops;
    std::unordered_set<size_t> retained_tensors;  // currently in fast memory

    for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
        const auto& sg = solution.subgraphs[sg_idx];

        for (auto op_id : sg.op_ids) {
            covered_ops.insert(op_id);
        }

        // Use the provided latency for now
        total += sg.subgraph_latency;

        // Update retained tensors
        retained_tensors.clear();
        for (auto t : sg.tensors_to_retain) {
            retained_tensors.insert(t);
        }
    }

    // Validate coverage
    for (size_t i = 0; i < problem.ops.size(); ++i) {
        if (covered_ops.find(i) == covered_ops.end()) {
            std::cerr << "WARNING: Op " << i << " not covered by any subgraph\n";
            return -1.0;
        }
    }

    return total;
}

}  // namespace mlsys
