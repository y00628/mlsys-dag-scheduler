#include <iostream>
#include <set>
#include <string>

#include "evaluator.h"
#include "problem.h"
#include "solution.h"
#include "solver.h"

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input.json> <output.json>\n";
        std::cerr << "       " << argv[0] << " <input.json> <solution_in.json> <output.json>  (eval-only)\n";
        return 1;
    }

    // eval-only mode: read existing solution, skip solver
    const bool eval_only = (argc == 4);
    std::string input_path  = argv[1];
    std::string eval_input  = eval_only ? argv[2] : "";
    std::string output_path = eval_only ? argv[3] : argv[2];

    try {
        // Parse the problem
        auto problem = mlsys::ReadProblem(input_path);
        std::cerr << "Loaded problem from: " << input_path << "\n";

        // Solve (or load pre-built solution)
        mlsys::Solution solution;
        if (eval_only) {
            solution = mlsys::ReadSolution(eval_input);
            std::cerr << "Loaded solution from: " << eval_input << "\n";
        } else {
            solution = mlsys::Solve(problem);
        }

        // Print solution summary.
        std::set<size_t> covered_ops;
        for (const auto& sg : solution.subgraphs) {
            covered_ops.insert(sg.op_ids.begin(), sg.op_ids.end());
        }
        std::cerr << "Solution: " << solution.subgraphs.size() << " subgraphs, "
                  << covered_ops.size() << "/" << problem.ops.size() << " ops covered\n";

        // Write output first (even if invalid — useful for debugging).
        mlsys::WriteSolution(solution, output_path);
        std::cerr << "Solution written to: " << output_path << "\n";

        // Validate and recompute subgraph latencies.
        mlsys::TotalLatency latency = -1.0;
        if (!mlsys::RecomputeLatencies(problem, &solution, &latency)) {
            std::cerr << "ERROR: Solution is invalid!\n";
            return 1;
        }
        std::cerr << "Total latency: " << latency << "\n";

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
