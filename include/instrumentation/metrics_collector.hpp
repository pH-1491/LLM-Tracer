#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <optional>

#include "core/layer_record.hpp"
#include "utils/math_utils.hpp"

namespace llmtracer {

struct TensorView {
    const float*             data     = nullptr;  // Raw float buffer (read-only)
    std::vector<int64_t>     shape;               // Dimension sizes, e.g. {1, 16, 768}
    std::string              name;                // Optional debug label

    // Total number of elements
    size_t numel() const {
        if (shape.empty() || !data) return 0;
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    TensorShape toTensorShape() const {
        return TensorShape(shape);
    }

    bool isValid() const { return data != nullptr && !shape.empty(); }
};



struct AttentionTensorView {
    const float* data      = nullptr;
    int          batch     = 0;       // Which batch item to sample (always 0)
    int          num_heads = 0;
    int          seq_len   = 0;

    // Total floats in the full tensor (all batches, all heads)
    size_t numel() const {
        return static_cast<size_t>(num_heads) *
               static_cast<size_t>(seq_len)   *
               static_cast<size_t>(seq_len);
    }

    // Pointer to the [num_heads, seq_len, seq_len] slice for batch index 0
    const float* batchPtr() const {
        return data + batch * numel();
    }

    // Pointer to the [seq_len, seq_len] matrix for a specific head
    const float* headPtr(int head) const {
        if (head < 0 || head >= num_heads) return nullptr;
        size_t head_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len);
        return batchPtr() + head * head_stride;
    }

    bool isValid() const {
        return data != nullptr && num_heads > 0 && seq_len > 0;
    }
};



struct CollectionConfig {
    // ── Activation stats ──────────────────────────────────────────────────────
    bool collect_input_stats  = true;   // Stats on the tensor entering each layer
    bool collect_output_stats = true;   // Stats on the tensor leaving each layer

    // ── Attention data ────────────────────────────────────────────────────────
    bool collect_attention    = true;   // Extract attention matrices
    int  max_attention_seq    = 64;     // Cap seq_len for heatmap (memory guard)
                                        // If seq > this, matrix is sub-sampled

    // ── Sparsity threshold ────────────────────────────────────────────────────
    float sparsity_threshold  = math::kSparsityThreshold;

    // ── Anomaly detection ─────────────────────────────────────────────────────
    bool flag_nan             = true;   // Detect NaN values
    bool flag_inf             = true;   // Detect Inf values
    bool flag_high_sparsity   = true;   // Warn when sparsity > 0.9

    // ── Performance ───────────────────────────────────────────────────────────
    // If true, stats computation runs on a background thread (lower hook latency
    // at the cost of slightly stale data in the TUI). If false, runs inline.
    bool async_stats          = false;  // Future extension; false = synchronous
};

class MetricsCollector {
public:
    explicit MetricsCollector(CollectionConfig cfg = {})
        : cfg_(std::move(cfg)) {}

    // ── Input tensor stats ────────────────────────────────────────────────────

    // Compute ActivationStats from the input tensor and store in record.input_stats.
    // Also stores the input shape in record.input_shape.
    void collectInputStats(const TensorView& tensor, LayerRecord& record) const {
        if (!cfg_.collect_input_stats) return;
        if (!tensor.isValid()) return;

        record.input_shape = tensor.toTensorShape();
        record.input_stats = computeStats(tensor);
    }

    // ── Output tensor stats ───────────────────────────────────────────────────

    // Compute ActivationStats from the output tensor and store in record.output_stats.
    // Also stores the output shape in record.output_shape.
    void collectOutputStats(const TensorView& tensor, LayerRecord& record) const {
        if (!cfg_.collect_output_stats) return;
        if (!tensor.isValid()) return;

        record.output_shape = tensor.toTensorShape();
        record.output_stats = computeStats(tensor);
    }

    // ── Attention matrix collection ───────────────────────────────────────────

    // Extract and store an attention matrix from an attention score tensor.
    // Populates record.attention with one head's matrix (default: head 0)
    // plus a mean-over-heads matrix that the TUI can switch between.
    //
    // If seq_len > cfg_.max_attention_seq, the matrix is sub-sampled
    // to keep memory usage bounded.
    void collectAttention(const AttentionTensorView& attn, LayerRecord& record) const {
        if (!cfg_.collect_attention) return;
        if (!attn.isValid())          return;

        // Determine effective seq_len after optional sub-sampling
        int eff_seq = attn.seq_len;
        bool subsampled = false;
        if (eff_seq > cfg_.max_attention_seq) {
            eff_seq    = cfg_.max_attention_seq;
            subsampled = true;
        }

        record.attention.num_heads  = attn.num_heads;
        record.attention.seq_len    = eff_seq;
        record.attention.head_shown = 0;  // Default to first head; TUI can change

        const size_t matrix_size = static_cast<size_t>(eff_seq * eff_seq);
        record.attention.matrix.resize(matrix_size);

        if (subsampled) {
            // Sub-sample: pick evenly spaced rows and cols from head 0
            extractSubsampledHead(attn, 0, eff_seq, record.attention.matrix);
        } else {
            // Full copy of head 0's matrix
            const float* src = attn.headPtr(0);
            if (src) {
                std::copy(src, src + matrix_size, record.attention.matrix.begin());
            }
        }

        // Normalize to [0,1] for display — raw logits / softmax'd weights both work
        math::normalizeToRange(record.attention.matrix.data(), matrix_size);
    }

    // Switch the displayed head in an already-collected AttentionData.
    // Called by the TUI when the user presses h/H to cycle heads.
    // Returns false if the head index is out of range or src data is unavailable.
    bool switchAttentionHead(const AttentionTensorView& attn,
                             int                         new_head,
                             AttentionData&              out) const
    {
        if (!attn.isValid()) return false;
        if (new_head < 0 || new_head >= attn.num_heads) return false;

        int eff_seq = std::min(attn.seq_len, cfg_.max_attention_seq);
        const size_t matrix_size = static_cast<size_t>(eff_seq * eff_seq);

        out.head_shown = new_head;
        out.seq_len    = eff_seq;
        out.matrix.resize(matrix_size);

        if (attn.seq_len > cfg_.max_attention_seq) {
            extractSubsampledHead(attn, new_head, eff_seq, out.matrix);
        } else {
            const float* src = attn.headPtr(new_head);
            if (!src) return false;
            std::copy(src, src + matrix_size, out.matrix.begin());
        }

        math::normalizeToRange(out.matrix.data(), matrix_size);
        return true;
    }

    // Compute and return a mean-over-all-heads attention matrix.
    // Used when the TUI switches to "average heads" view.
    std::vector<float> meanHeadsMatrix(const AttentionTensorView& attn) const {
        if (!attn.isValid()) return {};

        int eff_seq = std::min(attn.seq_len, cfg_.max_attention_seq);
        const size_t matrix_size = static_cast<size_t>(eff_seq * eff_seq);

        std::vector<float> result(matrix_size, 0.0f);

        if (attn.seq_len <= cfg_.max_attention_seq) {
            // Full resolution: use math utility directly
            std::vector<float> full(static_cast<size_t>(attn.num_heads) * matrix_size);
            const float* src = attn.batchPtr();
            std::copy(src, src + full.size(), full.begin());
            math::meanOverHeads(full.data(), result.data(), attn.num_heads, eff_seq);
        } else {
            // Sub-sample each head, then average
            std::vector<float> head_buf(matrix_size);
            for (int h = 0; h < attn.num_heads; ++h) {
                extractSubsampledHead(attn, h, eff_seq, head_buf);
                for (size_t i = 0; i < matrix_size; ++i) {
                    result[i] += head_buf[i];
                }
            }
            float inv = 1.0f / static_cast<float>(attn.num_heads);
            for (float& v : result) v *= inv;
        }

        math::normalizeToRange(result.data(), matrix_size);
        return result;
    }

    // ── Anomaly detection ─────────────────────────────────────────────────────

    // Check a LayerRecord's stats for anomalies and return a diagnostic string.
    // Empty string = no anomalies. Non-empty = description for TUI status bar.
    std::string diagnose(const LayerRecord& record) const {
        std::string msg;

        auto check = [&](const ActivationStats& s, const char* side) {
            if (cfg_.flag_nan && s.has_nan) {
                msg += std::string(side) + ": NaN detected! ";
            }
            if (cfg_.flag_inf && s.has_inf) {
                msg += std::string(side) + ": Inf detected! ";
            }
            if (cfg_.flag_high_sparsity && s.sparsity > 0.9f) {
                char buf[48];
                std::snprintf(buf, sizeof(buf),
                              "%s: %.0f%% sparse (possible dead neurons) ",
                              side, s.sparsity * 100.0f);
                msg += buf;
            }
        };

        check(record.input_stats,  "input");
        check(record.output_stats, "output");

        // Check for activation explosion (|max| > 1e4 is a common heuristic)
        auto checkClip = [&](const ActivationStats& s, const char* side) {
            if (std::abs(s.max_val) > 1e4f || std::abs(s.min_val) > 1e4f) {
                char buf[64];
                std::snprintf(buf, sizeof(buf),
                              "%s: large magnitude [%.1e, %.1e] (clipping risk) ",
                              side, static_cast<double>(s.min_val),
                              static_cast<double>(s.max_val));
                msg += buf;
            }
        };
        checkClip(record.input_stats,  "input");
        checkClip(record.output_stats, "output");

        return msg;
    }

    // ── Config access ─────────────────────────────────────────────────────────

    const CollectionConfig& config() const { return cfg_; }

    void setConfig(CollectionConfig cfg) { cfg_ = std::move(cfg); }

private:
    // ── Internal helpers ──────────────────────────────────────────────────────

    // Compute ActivationStats from a TensorView, respecting the config's
    // sparsity threshold and sampling cap.
    ActivationStats computeStats(const TensorView& tensor) const {
        return math::computeStats(tensor.data,
                                  tensor.numel(),
                                  cfg_.sparsity_threshold);
    }

    // Extract a sub-sampled [eff_seq × eff_seq] matrix from a specific head
    // of an attention tensor whose actual seq_len > eff_seq.
    // Uses stride sampling: picks evenly-spaced rows and columns.
    void extractSubsampledHead(const AttentionTensorView& attn,
                                int                        head,
                                int                        eff_seq,
                                std::vector<float>&        out) const
    {
        const float* src = attn.headPtr(head);
        if (!src) { std::fill(out.begin(), out.end(), 0.0f); return; }

        int orig_seq = attn.seq_len;
        double row_stride = static_cast<double>(orig_seq) / static_cast<double>(eff_seq);
        double col_stride = row_stride;

        for (int r = 0; r < eff_seq; ++r) {
            int src_row = static_cast<int>(r * row_stride);
            for (int c = 0; c < eff_seq; ++c) {
                int src_col = static_cast<int>(c * col_stride);
                out[static_cast<size_t>(r * eff_seq + c)] =
                    src[static_cast<size_t>(src_row * orig_seq + src_col)];
            }
        }
    }

    CollectionConfig cfg_;
};

}  // namespace llmtracer