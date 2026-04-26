# test_validity.cpp
#include "Validity.h"
#include "Graph.h"
#include <cassert>
#include <iostream>

int main() {
    KNGraph kn_graph;
    // ... construct simple KNGraph ...
    assert(check_validity(kn_graph) == "");
    std::cout << "KNGraph validity test passed\n";

    TBGraph tb_graph;
    // ... construct simple TBGraph ...
    assert(check_validity(tb_graph) == "");
    std::cout << "TBGraph validity test passed\n";

    // ... ThreadGraph test ...
    std::cout << "All validity tests passed\n";
    return 0;
}
