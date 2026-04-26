# test_expand_graph.cpp
#include "expand_graph.h"
#include "Graph.h"
#include <cassert>
#include <iostream>

int main() {
    KNGraph kn_graph;
    // ... construct simple KNGraph with GraphDefinedOperator ...
    ExpandedGraph expanded = expand_graph(kn_graph);
    // ... check expanded structure ...
    std::cout << "expand_graph test passed\n";
    return 0;
}
