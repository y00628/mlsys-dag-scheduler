# expand_graph.h
#pragma once
#include "Graph.h"
#include "GraphDefinedOperator.h"

// Recursively expands a KNGraph into a hierarchical structure
// (KNGraph -> TBGraph -> ThreadGraph)
class ExpandedGraph {
public:
    // ...
};

ExpandedGraph expand_graph(const KNGraph& kn_graph);
