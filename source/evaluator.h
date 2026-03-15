#pragma once

#include "problem.h"
#include "solution.h"

namespace mlsys {

// Evaluate a solution against a problem, returning total latency.
// Returns negative value on error (invalid solution).
TotalLatency Evaluate(const Problem& problem, const Solution& solution);

}  // namespace mlsys
