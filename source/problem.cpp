#include "problem.h"

#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

#include "../third_party/json.hpp"

using json = nlohmann::json;

namespace mlsys {

Problem ReadProblem(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    json j = json::parse(ifs);

    Problem problem;

    // Parse tensors from parallel arrays
    auto& widths = j["widths"];
    auto& heights = j["heights"];
    size_t num_tensors = widths.size();
    problem.tensors.resize(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
        problem.tensors[i].width = widths[i].get<Width>();
        problem.tensors[i].height = heights[i].get<Height>();
    }

    // Parse operations from parallel arrays
    auto& inputs = j["inputs"];
    auto& outputs = j["outputs"];
    auto& base_costs = j["base_costs"];
    auto& op_types = j["op_types"];
    size_t num_ops = op_types.size();
    problem.ops.resize(num_ops);
    for (size_t i = 0; i < num_ops; ++i) {
        problem.ops[i].op_type = op_types[i].get<std::string>();
        problem.ops[i].base_cost = base_costs[i].get<BaseCost>();
        for (auto& idx : inputs[i]) {
            problem.ops[i].inputs.push_back(idx.get<size_t>());
        }
        for (auto& idx : outputs[i]) {
            problem.ops[i].outputs.push_back(idx.get<size_t>());
        }
    }

    // Parse hardware spec
    problem.fast_memory_capacity = j["fast_memory_capacity"].get<FastMemoryCapacity>();
    problem.slow_memory_bandwidth = j["slow_memory_bandwidth"].get<SlowMemoryBandwidth>();

    auto& ng = j["native_granularity"];
    problem.native_granularity.width = ng[0].get<Width>();
    problem.native_granularity.height = ng[1].get<Height>();
    problem.native_granularity.depth = problem.native_granularity.width;  // default

    // Derive graph inputs and outputs
    std::set<size_t> produced;
    std::set<size_t> consumed;
    for (auto& op : problem.ops) {
        for (auto idx : op.outputs) produced.insert(idx);
        for (auto idx : op.inputs) consumed.insert(idx);
    }
    for (size_t i = 0; i < num_tensors; ++i) {
        if (produced.find(i) == produced.end()) {
            problem.graph_inputs.push_back(i);
        }
        if (consumed.find(i) == consumed.end()) {
            problem.graph_outputs.push_back(i);
        }
    }

    return problem;
}

}  // namespace mlsys
