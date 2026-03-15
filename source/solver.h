#pragma once

#include "problem.h"
#include "solution.h"

namespace mlsys {

// Solve a scheduling problem, returning the best solution found within
// the available time budget.
Solution Solve(const Problem& problem);

}  // namespace mlsys
