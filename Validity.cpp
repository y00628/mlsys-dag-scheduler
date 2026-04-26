# Validity.cpp
#include "Validity.h"
#include "Graph.h"
#include <sstream>

std::string check_validity(const KNGraph& graph) {
    // Example: check all tensors fit in device memory, all ops have valid inputs/outputs
    for (const auto& node : graph.nodes) {
        // ... check node input/output shapes, types ...
    }
    // ... check device memory constraints ...
    return ""; // valid
}

std::string check_validity(const TBGraph& graph) {
    // Example: check all tensors fit in shared memory, for-loop body structure, etc.
    for (const auto& node : graph.tb_nodes) {
        // ... check node input/output shapes, types ...
    }
    // ... check shared memory constraints ...
    return "";
}

std::string check_validity(const ThreadGraph& graph) {
    // Example: check all tensors fit in register file
    // ...
    return "";
}
