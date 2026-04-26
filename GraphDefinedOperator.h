# GraphDefinedOperator.h
#pragma once
#include "Graph.h"
#include <memory>

// Forward declaration
class TBGraph;

// Allows a KNNode to encapsulate a full TBGraph
class GraphDefinedOperator {
public:
    std::shared_ptr<TBGraph> tb_graph;
    GraphDefinedOperator(std::shared_ptr<TBGraph> tb) : tb_graph(tb) {}
};
