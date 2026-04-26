# Graph.h
#pragma once
#include <vector>
#include <string>
#include <memory>
#include "Tensor.h"

// Forward declarations
class TBGraph; // For future extension

// Kernel-level operator node
class KNNode {
public:
    std::string name; // e.g., "MatMul", "CustomOp"
    std::vector<int> input_tensor_ids;  // Indices into KNGraph::tensors
    std::vector<int> output_tensor_ids; // Indices into KNGraph::tensors
    // Optionally: operator type, attributes, etc.

    KNNode(const std::string& name_,
           const std::vector<int>& inputs,
           const std::vector<int>& outputs)
        : name(name_), input_tensor_ids(inputs), output_tensor_ids(outputs) {}
};

// Kernel-level edge (DTensor)
class KNEdge {
public:
    int tensor_id; // Index into KNGraph::tensors
    int producer_node; // Index into KNGraph::nodes
    std::vector<int> consumer_nodes; // Indices into KNGraph::nodes

    KNEdge(int tid, int prod, const std::vector<int>& consumers)
        : tensor_id(tid), producer_node(prod), consumer_nodes(consumers) {}
};

// Kernel-level graph
class KNGraph {
public:
    std::vector<KNNode> nodes;
    std::vector<KNEdge> edges;
    std::vector<std::shared_ptr<Tensor>> tensors; // All tensors in the graph

    // Add a node and return its index
    int add_node(const KNNode& node) {
        nodes.push_back(node);
        return nodes.size() - 1;
    }
    // Add a tensor and return its index
    int add_tensor(const std::shared_ptr<Tensor>& tensor) {
        tensors.push_back(tensor);
        return tensors.size() - 1;
    }
    // Add an edge and return its index
    int add_edge(const KNEdge& edge) {
        edges.push_back(edge);
        return edges.size() - 1;
    }
    // Get consumers of a tensor
    std::vector<int> get_consumers(int tensor_id) const {
        for (const auto& edge : edges) {
            if (edge.tensor_id == tensor_id) return edge.consumer_nodes;
        }
        return {};
    }
    // Get producer of a tensor
    int get_producer(int tensor_id) const {
        for (const auto& edge : edges) {
            if (edge.tensor_id == tensor_id) return edge.producer_node;
        }
        return -1;
    }
};
