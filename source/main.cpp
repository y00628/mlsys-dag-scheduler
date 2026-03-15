#include <iostream>
#include <string>

#include "evaluator.h"
#include "problem.h"
#include "solution.h"
#include "solver.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.json> <output.json>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    try {
        // Parse the problem
        auto problem = mlsys::ReadProblem(input_path);
        std::cerr << "Loaded problem from: " << input_path << "\n";

        // Solve
        auto solution = mlsys::Solve(problem);

        // Evaluate locally (optional sanity check)
        auto latency = mlsys::Evaluate(problem, solution);
        if (latency < 0) {
            std::cerr << "ERROR: Solution is invalid!\n";
        } else {
            std::cerr << "Total latency: " << latency << "\n";
        }

        // Write output
        mlsys::WriteSolution(solution, output_path);
        std::cerr << "Solution written to: " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
