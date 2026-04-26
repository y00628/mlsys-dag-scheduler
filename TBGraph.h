# TBGraph.h
#pragma once
#include "Graph.h"
#include <vector>
#include <string>
#include <memory>

// Threadblock-level operator node
class TBNode {
public:
    std::string name; // e.g., "InputIterator", "MatMul", "Accumulator"
    std::vector<int> input_tensor_ids;  // Indices into TBGraph::tensors
    std::vector<int> output_tensor_ids; // Indices into TBGraph::tensors
    // Optionally: operator type, attributes, etc.

    TBNode(const std::string& name_,
           const std::vector<int>& inputs,
           const std::vector<int>& outputs)
        : name(name_), input_tensor_ids(inputs), output_tensor_ids(outputs) {}
};

// Threadblock-level edge (STensor)
class TBEdge {
public:
    int tensor_id; // Index into TBGraph::tensors
    int producer_node; // Index into TBGraph::tb_nodes
    std::vector<int> consumer_nodes; // Indices into TBGraph::tb_nodes

    TBEdge(int tid, int prod, const std::vector<int>& consumers)
        : tensor_id(tid), producer_node(prod), consumer_nodes(consumers) {}
};

// Threadblock-level graph (inherits from KNGraph)
class TBGraph : public KNGraph {
public:
    std::vector<TBNode> tb_nodes;
    std::vector<TBEdge> tb_edges;
    std::vector<std::shared_ptr<Tensor>> tensors; // All tensors in the TBGraph
    std::vector<int> grid_dim;      // e.g., {grid_x, grid_y, grid_z}
    std::vector<int> for_loop_dim;  // e.g., {k_split}

    // Add a TBNode and return its index
    int add_tbnode(const TBNode& node) {
        tb_nodes.push_back(node);
        return tb_nodes.size() - 1;
    }
    // Add a TBEdge and return its index
    int add_tbedge(const TBEdge& edge) {
        tb_edges.push_back(edge);
        return tb_edges.size() - 1;
    }
};
