# Validity.h
#pragma once
#include "Graph.h"
#include <vector>
#include <string>

// Forward declarations
class KNGraph;
class TBGraph;
class ThreadGraph;

// Returns empty string if valid, else error message
std::string check_validity(const KNGraph& graph);
std::string check_validity(const TBGraph& graph);
std::string check_validity(const ThreadGraph& graph);
