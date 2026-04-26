# Tensor.h
#pragma once
#include <vector>
#include <string>
#include <optional>

// Enum for memory type
enum class MemoryType {
    Global,
    Shared,
    Register
};


// Tensor base class
class Tensor {
public:
    std::vector<int> shape;        // e.g., {M, N}
    std::string dtype;             // e.g., "float16", "float32"
    std::string layout;            // e.g., "row-major", "col-major"
    MemoryType memory_type;        // Global/Shared/Register
    std::optional<std::string> swizzle; // For STensor only, e.g., "none", "xor", "shift"
    std::vector<int> stride;       // stride for each dim (optional, for advanced layout)

    Tensor(const std::vector<int>& shape_,
           const std::string& dtype_,
           const std::string& layout_,
           MemoryType memory_type_,
           std::optional<std::string> swizzle_ = std::nullopt)
        : shape(shape_), dtype(dtype_), layout(layout_), memory_type(memory_type_), swizzle(swizzle_) {
        stride = calc_tensor_strides(shape_);
    }

    // Utility: get total number of elements
    int num_elements() const {
        int n = 1;
        for (int d : shape) n *= d;
        return n;
    }

    // Utility: calculate stride for each dimension (row-major by default)
    static std::vector<int> calc_tensor_strides(const std::vector<int>& shape) {
        std::vector<int> stride(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        return stride;
    }

    // Utility: find innermost (contiguous) dimension
    int find_innermost_dim() const {
        if (stride.empty()) return -1;
        int min_stride = stride[0];
        int min_idx = 0;
        for (int i = 1; i < stride.size(); ++i) {
            if (stride[i] < min_stride) {
                min_stride = stride[i];
                min_idx = i;
            }
        }
        return min_idx;
    }
};

// DTensor: Device memory tensor (inherits Tensor)
class DTensor : public Tensor {
public:
    DTensor(const std::vector<int>& shape_,
            const std::string& dtype_,
            const std::string& layout_)
        : Tensor(shape_, dtype_, layout_, MemoryType::Global) {}
};

// STensor: Shared memory tensor (inherits Tensor)
class STensor : public Tensor {
public:
    STensor(const std::vector<int>& shape_,
            const std::string& dtype_,
            const std::string& layout_,
            const std::string& swizzle_ = "none")
        : Tensor(shape_, dtype_, layout_, MemoryType::Shared, swizzle_) {}
};

// Register tensor (optional, for thread-level)
class RegTensor : public Tensor {
public:
    RegTensor(const std::vector<int>& shape_,
              const std::string& dtype_,
              const std::string& layout_)
        : Tensor(shape_, dtype_, layout_, MemoryType::Register) {}
};
