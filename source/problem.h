#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlsys {

// Type aliases matching the official mlsys.h
using Width = int64_t;
using Height = int64_t;
using Depth = int64_t;
using BaseCost = int64_t;
using FastMemoryCapacity = int64_t;
using SlowMemoryBandwidth = int64_t;

struct Tensor {
    Width width;
    Height height;
};

struct Op {
    std::string op_type;           // "MatMul" or "Pointwise"
    std::vector<size_t> inputs;    // tensor indices consumed (for MatMul: [Left, Right])
    std::vector<size_t> outputs;   // tensor indices produced
    BaseCost base_cost;
};

struct Granularity {
    Width width;
    Height height;
    Depth depth;                   // reduction dimension (k); ignored for Pointwise
};

struct Problem {
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    FastMemoryCapacity fast_memory_capacity;
    SlowMemoryBandwidth slow_memory_bandwidth;
    Granularity native_granularity;  // [w, h] from JSON; depth defaults to native

    // Derived: which tensors are graph inputs/outputs
    std::vector<size_t> graph_inputs;   // tensors not produced by any op
    std::vector<size_t> graph_outputs;  // tensors not consumed by any op
};

// Parse a problem JSON file into a Problem struct.
Problem ReadProblem(const std::string& filename);

}  // namespace mlsys
