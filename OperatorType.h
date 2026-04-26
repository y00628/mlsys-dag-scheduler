# OperatorType.h
#pragma once
#include <string>

// Kernel Node (KN) operator types
enum class KNOperatorType {
    MatMul,
    Pointwise,
    Unknown
};

inline std::string KNOperatorTypeToString(KNOperatorType type) {
    switch (type) {
        case KNOperatorType::MatMul: return "MatMul";
        case KNOperatorType::Pointwise: return "Pointwise";
        default: return "Unknown";
    }
}

// Thread Block (TB) operator types
enum class TBOperatorType {
    MatMul,
    Pointwise,
    Unknown
};

inline std::string TBOperatorTypeToString(TBOperatorType type) {
    switch (type) {
        case TBOperatorType::MatMul: return "MatMul";
        case TBOperatorType::Pointwise: return "Pointwise";
        default: return "Unknown";
    }
}
