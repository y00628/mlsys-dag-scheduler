# ThreadTensor.h
#pragma once
#include "Tensor.h"
#include <vector>
#include <string>

// Thread-level tensor (register)
class ThreadTensor : public Tensor {
public:
    ThreadTensor(const std::vector<int>& shape_,
                 const std::string& dtype_,
                 const std::string& layout_)
        : Tensor(shape_, dtype_, layout_, MemoryType::Register) {}
};
