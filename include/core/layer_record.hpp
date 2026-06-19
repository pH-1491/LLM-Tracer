#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace llmtracer {

enum class LayerType {
    Embedding,      // Token + positional embedding lookup
    Attention,      // Multi-head self-attention (QKV + softmax + proj)
    MLP,            // Feed-forward / MLP block
    LayerNorm,      // Pre/post normalization
    Output,         // Final logit projection
    Unknown         // Fallback for unlabelled hooks
};

// Convert enum to a short display string (used in TUI layer list)
inline const char* layerTypeToString(LayerType t) {
    switch (t) {
        case LayerType::Embedding:  return "Embed";
        case LayerType::Attention:  return "Attn ";
        case LayerType::MLP:        return "MLP  ";
        case LayerType::LayerNorm:  return "Norm ";
        case LayerType::Output:     return "Out  ";
        default:                    return "Unk  ";
    }
}


struct TensorShape {
    std::vector<int64_t> dims;  // Ordered list of dimension sizes

    TensorShape() = default;
    explicit TensorShape(std::vector<int64_t> d) : dims(std::move(d)) {}

    // Total number of elements (product of all dims)
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }


    std::string toString() const {
        if (dims.empty()) return "[]";
        std::string s = "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            s += std::to_string(dims[i]);
            if (i + 1 < dims.size()) s += ", ";
        }
        s += "]";
        return s;
    }
};




struct ActivationStats {
    float mean       = 0.0f;   // Arithmetic mean of all values
    float std_dev    = 0.0f;   // Standard deviation
    float min_val    = 0.0f;   // Minimum value
    float max_val    = 0.0f;   // Maximum value
    float sparsity   = 0.0f;   // Fraction of values ~= 0.0 (|x| < threshold)
    bool  has_nan    = false;   // True if any NaN detected
    bool  has_inf    = false;   // True if any Inf detected

    // Flags anomalies worth highlighting in the TUI
    bool isAnomolous() const { return has_nan || has_inf || sparsity > 0.9f; }
};




struct AttentionData {
    int                          num_heads  = 0;
    int                          seq_len    = 0;
    int                          head_shown = 0;  // Which head is currently displayed


    std::vector<float>           matrix;

    bool isValid() const { return !matrix.empty() && seq_len > 0; }

    // Access element at (row, col) safely
    float at(int row, int col) const {
        if (row < 0 || col < 0 || row >= seq_len || col >= seq_len) return 0.0f;
        return matrix[static_cast<size_t>(row * seq_len + col)];
    }
};




struct LayerRecord {
    // ── Identity ──────────────────────────────────────────────────────────────
    uint64_t    record_id   = 0;        // Monotonically increasing ID
    std::string layer_name;             // e.g. "transformer.h.3.attn"
    LayerType   layer_type  = LayerType::Unknown;
    int         layer_index = -1;       // Position in the overall layer sequence



    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

    // Convenience: duration in microseconds
    int64_t latencyMicros() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
    }


    TensorShape input_shape;    // Shape of the tensor entering this layer
    TensorShape output_shape;   // Shape of the tensor leaving this layer


    ActivationStats input_stats;
    ActivationStats output_stats;


    AttentionData attention;


    LayerRecord() = default;

    LayerRecord(uint64_t id, std::string name, LayerType type, int idx)
        : record_id(id)
        , layer_name(std::move(name))
        , layer_type(type)
        , layer_index(idx)
    {}
};

}