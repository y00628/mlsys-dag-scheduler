#pragma once

#include "problem.h"
#include "solution.h"

namespace mlsys {

struct SubgraphEvaluation {
    bool valid = false;
    SubgraphLatency latency = -1.0;
    int64_t peak_live_bytes = 0;
};

// Evaluate a solution against a problem, returning total latency.
// Returns negative value on error (invalid solution).
TotalLatency Evaluate(const Problem& problem, const Solution& solution);

// Evaluates a single subgraph using the same simulation core as the full
// evaluator. `retained_inputs` are tensors kept resident from the previous
// subgraph and available to this subgraph if they are boundary inputs.
bool EvaluateSubgraph(const Problem& problem, const Subgraph& subgraph,
                      const std::vector<size_t>& retained_inputs,
                      SubgraphEvaluation* evaluation);

// Recomputes subgraph latencies in-place using the evaluator logic.
// Returns false if the solution is invalid.
bool RecomputeLatencies(const Problem& problem, Solution* solution,
                        TotalLatency* total_latency);

}  // namespace mlsys
