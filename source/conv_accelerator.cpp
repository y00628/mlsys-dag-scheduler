#include "conv_accelerator.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mlsys {

namespace {

constexpr int64_t kInvalidCost = std::numeric_limits<int64_t>::max() / 4;

int64_t NormalizeChannelTile(int64_t requested, int64_t full_channels) {
    if (full_channels <= 0) {
        return 0;
    }
    if (requested <= 0) {
        return full_channels;
    }
    return std::min(requested, full_channels);
}

int64_t ActivationTileBytes(int64_t height, int64_t width, int64_t channels,
                            int64_t bytes_per_element) {
    return BytesForElements(height * width * channels, bytes_per_element);
}

int64_t FullInputBytes(const Conv2DOp& op) {
    return BytesForElements(op.input_height * op.input_width * op.input_channels,
                            op.bytes_per_element);
}

int64_t FullOutputBytes(const Conv2DOp& op) {
    return BytesForElements(op.output_height * op.output_width * op.output_channels,
                            op.bytes_per_element);
}

}  // namespace

int64_t CeilDiv(int64_t numerator, int64_t denominator) {
    if (denominator <= 0) {
        throw std::invalid_argument("CeilDiv requires denominator > 0");
    }
    if (numerator <= 0) {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}

int64_t ComputeConvOutputDim(int64_t input, int64_t kernel, int64_t stride) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || input < kernel) {
        return 0;
    }
    return (input - kernel) / stride + 1;
}

int64_t ComputeRequiredInputDim(int64_t output_tile, int64_t kernel,
                                int64_t stride) {
    if (output_tile <= 0 || kernel <= 0 || stride <= 0) {
        return 0;
    }
    return output_tile * stride + kernel - stride;
}

int64_t BytesForElements(int64_t element_count, int64_t bytes_per_element) {
    if (element_count <= 0 || bytes_per_element <= 0) {
        return 0;
    }
    return element_count * bytes_per_element;
}

int64_t ConvParameterCount(const Conv2DOp& op) {
    return op.output_channels * op.input_channels * op.kernel_height *
           op.kernel_width;
}

int64_t ConvParameterBytes(const Conv2DOp& op) {
    return BytesForElements(ConvParameterCount(op), op.bytes_per_element);
}

int64_t EffectivePEThroughput(const ConvAcceleratorSpec& spec) {
    if (spec.pe_throughput > 0) {
        return spec.pe_throughput;
    }

    const int64_t rows = std::max<int64_t>(1, spec.pe_array_rows);
    const int64_t cols = std::max<int64_t>(1, spec.pe_array_cols);

    switch (spec.dataflow) {
        case ConvDataflow::kOutputStationary:
            return cols;
        case ConvDataflow::kWeightStationary:
            return rows;
        case ConvDataflow::kInputStationary:
            return rows;
        case ConvDataflow::kRowStationary:
            return std::max<int64_t>(1, std::min(rows, cols));
        case ConvDataflow::kUnknown:
        default:
            return std::max<int64_t>(1, rows * cols);
    }
}

Conv2DOp NormalizeConv2DOp(const Conv2DOp& op) {
    Conv2DOp normalized = op;

    if (normalized.stride_height <= 0) normalized.stride_height = 1;
    if (normalized.stride_width <= 0) normalized.stride_width = 1;
    if (normalized.kernel_height <= 0) normalized.kernel_height = 1;
    if (normalized.kernel_width <= 0) normalized.kernel_width = 1;
    if (normalized.bytes_per_element <= 0) normalized.bytes_per_element = 2;

    if (normalized.output_height <= 0) {
        normalized.output_height = ComputeConvOutputDim(
            normalized.input_height, normalized.kernel_height,
            normalized.stride_height);
    }
    if (normalized.output_width <= 0) {
        normalized.output_width = ComputeConvOutputDim(
            normalized.input_width, normalized.kernel_width,
            normalized.stride_width);
    }

    return normalized;
}

std::vector<ConvTileShape> PropagateOutputTilesBackward(
    const std::vector<Conv2DOp>& ops, const ConvTileShape& terminal_tile) {
    if (ops.empty()) {
        return {};
    }

    std::vector<Conv2DOp> normalized;
    normalized.reserve(ops.size());
    for (const auto& op : ops) {
        normalized.push_back(NormalizeConv2DOp(op));
    }

    std::vector<ConvTileShape> per_op(normalized.size());
    per_op.back() = terminal_tile;
    per_op.back().output_height =
        std::min(per_op.back().output_height, normalized.back().output_height);
    per_op.back().output_width =
        std::min(per_op.back().output_width, normalized.back().output_width);
    per_op.back().output_channels =
        NormalizeChannelTile(per_op.back().output_channels,
                             normalized.back().output_channels);

    for (int i = static_cast<int>(normalized.size()) - 2; i >= 0; --i) {
        const Conv2DOp& consumer = normalized[i + 1];
        ConvTileShape shape;
        shape.output_height = ComputeRequiredInputDim(
            per_op[i + 1].output_height, consumer.kernel_height,
            consumer.stride_height);
        shape.output_width = ComputeRequiredInputDim(
            per_op[i + 1].output_width, consumer.kernel_width,
            consumer.stride_width);
        shape.output_channels = normalized[i].output_channels;
        per_op[i] = shape;
    }

    return per_op;
}

ISOADecision DecideISOAForChain(const std::vector<Conv2DOp>& ops,
                                const ConvAcceleratorSpec& spec,
                                const ConvTileShape& terminal_tile) {
    ISOADecision decision;
    if (ops.empty()) {
        return decision;
    }

    std::vector<Conv2DOp> normalized;
    normalized.reserve(ops.size());
    for (const auto& op : ops) {
        normalized.push_back(NormalizeConv2DOp(op));
    }

    const auto tiles = PropagateOutputTilesBackward(normalized, terminal_tile);
    const int64_t throughput = std::max<int64_t>(1, EffectivePEThroughput(spec));

    const int n = static_cast<int>(normalized.size());
    std::vector<int64_t> dp_yes(n, kInvalidCost);
    std::vector<int64_t> dp_no(n, kInvalidCost);

    for (int i = n - 1; i >= 0; --i) {
        const auto& op = normalized[i];
        const auto& tile = tiles[i];
        const int64_t bytes = op.bytes_per_element;

        const int64_t isoa_channels =
            std::min<int64_t>(throughput, tile.output_channels);
        dp_yes[i] = ActivationTileBytes(tile.output_height, tile.output_width,
                                        isoa_channels, bytes);
        dp_no[i] = ActivationTileBytes(tile.output_height, tile.output_width,
                                       tile.output_channels, bytes);

        if (i + 1 < n) {
            dp_yes[i] += dp_no[i + 1];
            dp_no[i] += std::min(dp_no[i + 1], dp_yes[i + 1]);
        }
    }

    decision.apply_isoa.assign(n, false);
    decision.minimum_activation_bytes = std::min(dp_no[0], dp_yes[0]);

    bool must_disable_isoa = false;
    for (int i = 0; i < n; ++i) {
        bool use_isoa = false;
        if (spec.supports_isoa && !must_disable_isoa) {
            use_isoa = (dp_yes[i] <= dp_no[i]);
        }
        decision.apply_isoa[i] = use_isoa;
        must_disable_isoa = use_isoa;
        if (!use_isoa) {
            must_disable_isoa = false;
        }
    }

    return decision;
}

ConvGroupScheduleEstimate AnalyzeConvChain(const std::vector<Conv2DOp>& ops,
                                           const ConvAcceleratorSpec& spec,
                                           const ConvTileShape& terminal_tile) {
    ConvGroupScheduleEstimate estimate;
    estimate.tile = terminal_tile;

    if (ops.empty()) {
        estimate.invalid_reason = "Conv chain is empty";
        return estimate;
    }

    std::vector<Conv2DOp> normalized;
    normalized.reserve(ops.size());
    for (const auto& op : ops) {
        normalized.push_back(NormalizeConv2DOp(op));
    }

    const Conv2DOp& last = normalized.back();
    if (terminal_tile.output_height <= 0 || terminal_tile.output_width <= 0) {
        estimate.invalid_reason = "Terminal tile must be positive";
        return estimate;
    }
    if (terminal_tile.output_height > last.output_height ||
        terminal_tile.output_width > last.output_width) {
        estimate.invalid_reason = "Terminal tile exceeds output dimensions";
        return estimate;
    }

    const auto per_op_tiles =
        PropagateOutputTilesBackward(normalized, terminal_tile);
    estimate.isoa = DecideISOAForChain(normalized, spec, terminal_tile);

    estimate.per_op.reserve(normalized.size());
    const int64_t throughput = std::max<int64_t>(1, EffectivePEThroughput(spec));

    int64_t total_activation_bytes = 0;
    int64_t total_parameter_bytes = 0;
    int64_t total_line_buffer_bytes = 0;

    for (size_t i = 0; i < normalized.size(); ++i) {
        const auto& op = normalized[i];
        const auto& tile = per_op_tiles[i];
        if (tile.output_height > op.output_height ||
            tile.output_width > op.output_width) {
            estimate.invalid_reason =
                "Backward-propagated tile exceeds operator output dimensions";
            return estimate;
        }
        const bool use_isoa =
            i < estimate.isoa.apply_isoa.size() && estimate.isoa.apply_isoa[i];

        const int64_t channel_tile = use_isoa
                                         ? std::min<int64_t>(throughput,
                                                             tile.output_channels)
                                         : tile.output_channels;

        int64_t line_buffer_bytes = 0;
        if (spec.supports_line_buffer && !use_isoa) {
            const int64_t overlap_rows =
                std::max<int64_t>(0, op.kernel_height - op.stride_height);
            line_buffer_bytes = BytesForElements(
                overlap_rows * tile.output_width * channel_tile,
                op.bytes_per_element);
        }

        ConvOpFootprint footprint;
        footprint.required_output_height = tile.output_height;
        footprint.required_output_width = tile.output_width;
        footprint.channel_tile = channel_tile;
        footprint.activation_bytes = ActivationTileBytes(
            tile.output_height, tile.output_width, channel_tile,
            op.bytes_per_element);
        footprint.parameter_bytes = ConvParameterBytes(op);
        footprint.line_buffer_bytes = line_buffer_bytes;
        estimate.per_op.push_back(footprint);

        total_activation_bytes += footprint.activation_bytes;
        total_parameter_bytes += footprint.parameter_bytes;
        total_line_buffer_bytes += footprint.line_buffer_bytes;
    }

    if (spec.line_buffer_bytes > 0 &&
        total_line_buffer_bytes > spec.line_buffer_bytes) {
        estimate.invalid_reason = "Line buffer requirement exceeds capacity";
        return estimate;
    }

    estimate.working_set.activation_bytes = total_activation_bytes;
    estimate.working_set.parameter_bytes = total_parameter_bytes;
    estimate.working_set.line_buffer_bytes = total_line_buffer_bytes;
    estimate.working_set.total_bytes = estimate.working_set.activation_bytes +
                                       estimate.working_set.parameter_bytes +
                                       estimate.working_set.line_buffer_bytes;

    estimate.subgroup_count =
        CeilDiv(last.output_height, terminal_tile.output_height) *
        CeilDiv(last.output_width, terminal_tile.output_width);

    const bool parameters_fit_once =
        estimate.working_set.total_bytes <= spec.on_chip_buffer_bytes;
    estimate.parameter_refills =
        parameters_fit_once ? 1 : std::max<int64_t>(1, estimate.subgroup_count);

    estimate.estimated_memory_traffic_bytes =
        static_cast<double>(FullInputBytes(normalized.front()) +
                            FullOutputBytes(normalized.back())) +
        static_cast<double>(estimate.parameter_refills) *
            static_cast<double>(total_parameter_bytes);

    if (estimate.working_set.total_bytes > spec.on_chip_buffer_bytes) {
        estimate.invalid_reason = "Working set exceeds on-chip buffer";
        return estimate;
    }

    estimate.valid = true;
    return estimate;
}

}  // namespace mlsys
