#pragma once

#include "problem.h"
#include "solution.h"

namespace mlsys {

// Evaluate a solution against a problem, returning total latency.
// Returns negative value on error (invalid solution).
TotalLatency Evaluate(const Problem& problem, const Solution& solution);

// Recomputes subgraph latencies in-place using the evaluator logic.
// Returns false if the solution is invalid.
bool RecomputeLatencies(const Problem& problem, Solution* solution,
                        TotalLatency* total_latency);

}  // namespace mlsys
