#pragma once

#include "problem.h"

#include <optional>
#include <string>
#include <vector>

namespace mlsys {

using TraversalOrder = std::vector<int64_t>;
using SubgraphLatency = double;
using TotalLatency = double;

struct Subgraph {
    std::vector<size_t> op_ids;
    std::vector<size_t> tensors_to_retain;
    Granularity granularity;
    std::optional<TraversalOrder> traversal_order;
    SubgraphLatency subgraph_latency = 0.0;
};

struct Solution {
    std::vector<Subgraph> subgraphs;
};

// Write a solution to a JSON file.
void WriteSolution(const Solution& solution, const std::string& filename);

}  // namespace mlsys
