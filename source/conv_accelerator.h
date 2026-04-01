#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlsys {

enum class ConvDataflow {
    kUnknown = 0,
    kOutputStationary,
    kWeightStationary,
    kInputStationary,
    kRowStationary,
};

struct Conv2DOp {
    std::string name;

    int64_t input_height = 0;
    int64_t input_width = 0;
    int64_t input_channels = 0;
    int64_t output_channels = 0;

    int64_t kernel_height = 1;
    int64_t kernel_width = 1;
    int64_t stride_height = 1;
    int64_t stride_width = 1;

    // If omitted (<= 0), they are derived from input/kernel/stride.
    int64_t output_height = 0;
    int64_t output_width = 0;

    int64_t bytes_per_element = 2;
};

struct ConvTileShape {
    int64_t output_height = 1;
    int64_t output_width = 1;
    int64_t output_channels = 0;  // 0 => full output channel tile
};

struct ConvAcceleratorSpec {
    int64_t on_chip_buffer_bytes = 0;
    int64_t line_buffer_bytes = 0;

    int64_t pe_array_rows = 1;
    int64_t pe_array_cols = 1;
    int64_t rf_bytes_per_pe = 0;

    // If set to <= 0, a heuristic value is derived from dataflow and PE shape.
    int64_t pe_throughput = 0;

    ConvDataflow dataflow = ConvDataflow::kUnknown;
    bool supports_line_buffer = true;
    bool supports_isoa = true;
};

struct ConvOpFootprint {
    int64_t required_output_height = 0;
    int64_t required_output_width = 0;
    int64_t channel_tile = 0;
    int64_t activation_bytes = 0;
    int64_t parameter_bytes = 0;
    int64_t line_buffer_bytes = 0;
};

struct ISOADecision {
    std::vector<bool> apply_isoa;
    int64_t minimum_activation_bytes = 0;
};

struct ConvWorkingSetBreakdown {
    int64_t activation_bytes = 0;
    int64_t parameter_bytes = 0;
    int64_t line_buffer_bytes = 0;
    int64_t total_bytes = 0;
};

struct ConvGroupScheduleEstimate {
    bool valid = false;
    std::string invalid_reason;

    ConvTileShape tile;
    ISOADecision isoa;
    std::vector<ConvOpFootprint> per_op;
    ConvWorkingSetBreakdown working_set;

    int64_t subgroup_count = 0;
    int64_t parameter_refills = 0;
    double estimated_memory_traffic_bytes = 0.0;
};

int64_t CeilDiv(int64_t numerator, int64_t denominator);
int64_t ComputeConvOutputDim(int64_t input, int64_t kernel, int64_t stride);
int64_t ComputeRequiredInputDim(int64_t output_tile, int64_t kernel, int64_t stride);
int64_t BytesForElements(int64_t element_count, int64_t bytes_per_element);
int64_t ConvParameterCount(const Conv2DOp& op);
int64_t ConvParameterBytes(const Conv2DOp& op);
int64_t EffectivePEThroughput(const ConvAcceleratorSpec& spec);
Conv2DOp NormalizeConv2DOp(const Conv2DOp& op);

// Back-propagates the output tile of the last fused op to the earlier ops.
std::vector<ConvTileShape> PropagateOutputTilesBackward(
    const std::vector<Conv2DOp>& ops, const ConvTileShape& terminal_tile);

// Dynamic-programming decision for the ISOA pattern on a linear fused chain.
ISOADecision DecideISOAForChain(const std::vector<Conv2DOp>& ops,
                                const ConvAcceleratorSpec& spec,
                                const ConvTileShape& terminal_tile);

// Estimates buffer footprint and traffic for a linear fused Conv chain.
ConvGroupScheduleEstimate AnalyzeConvChain(const std::vector<Conv2DOp>& ops,
                                           const ConvAcceleratorSpec& spec,
                                           const ConvTileShape& terminal_tile);

}  // namespace mlsys
