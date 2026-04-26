#include "solution.h"

#include <fstream>
#include <stdexcept>

#include "../third_party/json.hpp"

using json = nlohmann::json;

namespace mlsys {

void WriteSolution(const Solution& solution, const std::string& filename) {
    json j;

    // Build parallel arrays for the solution
    json op_ids_arr = json::array();
    json tensors_to_retain_arr = json::array();
    json granularities_arr = json::array();
    json traversal_orders_arr = json::array();
    json subgraph_latencies_arr = json::array();

    for (const auto& sg : solution.subgraphs) {
        op_ids_arr.push_back(sg.op_ids);
        tensors_to_retain_arr.push_back(sg.tensors_to_retain);
        granularities_arr.push_back({sg.granularity.width,
                                     sg.granularity.height,
                                     sg.granularity.depth});

        if (sg.traversal_order.has_value()) {
            traversal_orders_arr.push_back(sg.traversal_order.value());
        } else {
            traversal_orders_arr.push_back(nullptr);
        }

        subgraph_latencies_arr.push_back(sg.subgraph_latency);
    }

    j["subgraphs"] = op_ids_arr;
    j["tensors_to_retain"] = tensors_to_retain_arr;
    j["granularities"] = granularities_arr;
    j["traversal_orders"] = traversal_orders_arr;
    j["subgraph_latencies"] = subgraph_latencies_arr;

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }
    ofs << j.dump(2) << std::endl;
}

Solution ReadSolution(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open solution file: " + filename);
    }
    json j;
    ifs >> j;

    Solution sol;
    const auto& subgraphs = j["subgraphs"];
    const auto& retains   = j["tensors_to_retain"];
    const auto& grans     = j["granularities"];
    const auto& travs     = j["traversal_orders"];
    const auto& lats      = j["subgraph_latencies"];
    for (size_t i = 0; i < subgraphs.size(); ++i) {
        Subgraph sg;
        sg.op_ids = subgraphs[i].get<std::vector<size_t>>();
        sg.tensors_to_retain = retains[i].get<std::vector<size_t>>();
        const auto& g = grans[i];
        sg.granularity.width  = g[0].get<int64_t>();
        sg.granularity.height = g[1].get<int64_t>();
        sg.granularity.depth  = g[2].get<int64_t>();
        if (!travs[i].is_null()) {
            sg.traversal_order = travs[i].get<std::vector<int64_t>>();
        }
        sg.subgraph_latency = lats[i].get<double>();
        sol.subgraphs.push_back(std::move(sg));
    }
    return sol;
}

}  // namespace mlsys
