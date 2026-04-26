# TensorMapping.h
#pragma once
#include <vector>
#include <string>
#include <optional>

// imap: Input Map (block id to tensor region)
class InputMap {
public:
    // Example: block_id -> (start_row, start_col, ...)
    std::vector<int> block_shape; // e.g., {tile_h, tile_w}
    InputMap(const std::vector<int>& block_shape_) : block_shape(block_shape_) {}
    // Given block_id, return offset for each dim
    std::vector<int> apply(int block_id) const {
        // For 2D: block_id = by * num_blocks_x + bx
        int bx = block_id % block_shape[1];
        int by = block_id / block_shape[1];
        return {by * block_shape[0], bx * block_shape[1]};
    }
};

// omap: Output Map (block id to output region)
class OutputMap {
public:
    std::vector<int> block_shape;
    OutputMap(const std::vector<int>& block_shape_) : block_shape(block_shape_) {}
    std::vector<int> apply(int block_id) const {
        // Same as InputMap for most cases
        int bx = block_id % block_shape[1];
        int by = block_id / block_shape[1];
        return {by * block_shape[0], bx * block_shape[1]};
    }
};

// fmap: For-loop Map (loop index to tensor region)
class ForLoopMap {
public:
    int loop_stride;
    ForLoopMap(int loop_stride_) : loop_stride(loop_stride_) {}
    int apply(int loop_idx) const {
        return loop_idx * loop_stride;
    }
};
