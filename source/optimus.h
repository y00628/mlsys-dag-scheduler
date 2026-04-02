#pragma once

#include "problem.h"
#include "solution.h"

namespace mlsys {

// Optimus-inspired solver backend for the contest problem model.
Solution SolveWithOptimus(const Problem& problem);
Solution SolveWithOptimusConvGuidance(const Problem& problem);
Solution SolveWithOptimusConvRerankV2(const Problem& problem);
Solution SolveWithPaperOptimus(const Problem& problem);

}  // namespace mlsys
