#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdint>
#include <cassert>

#include "core/layer_record.hpp"

namespace llmtracer {
namespace math {


constexpr float kSparsityThreshold = 1e-6f;

constexpr size_t kMaxStatSamples = 65536;

namespace detail {

// Returns indices into the source array, evenly spaced.
inline std::vector<size_t> sampleIndices(size_t total, size_t max_samples) {
    std::vector<size_t> indices;
    if (total == 0) return indices;

    if (total <= max_samples) {
        indices.resize(total);
        std::iota(indices.begin(), indices.end(), 0);
    } else {
        indices.reserve(max_samples);
        double stride = static_cast<double>(total) / static_cast<double>(max_samples);
        for (size_t i = 0; i < max_samples; ++i) {
            indices.push_back(static_cast<size_t>(i * stride));
        }
    }
    return indices;
}

}




inline ActivationStats computeStats(const float* data,
                                    size_t        count,
                                    float         threshold = kSparsityThreshold)
{
    ActivationStats stats{};
    if (!data || count == 0) return stats;


    auto indices = detail::sampleIndices(count, kMaxStatSamples);
    const size_t n = indices.size();


    double sum        = 0.0;
    float  min_val    = std::numeric_limits<float>::max();
    float  max_val    = std::numeric_limits<float>::lowest();
    size_t zero_count = 0;
    bool   has_nan    = false;
    bool   has_inf    = false;

    for (size_t idx : indices) {
        float v = data[idx];

        if (std::isnan(v)) { has_nan = true; continue; }
        if (std::isinf(v)) { has_inf = true; continue; }

        sum += static_cast<double>(v);
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        if (std::abs(v) < threshold) ++zero_count;
    }


    size_t valid = n;
    if (has_nan || has_inf) {
        valid = 0;
        for (size_t idx : indices) {
            float v = data[idx];
            if (!std::isnan(v) && !std::isinf(v)) ++valid;
        }
    }

    stats.has_nan = has_nan;
    stats.has_inf = has_inf;

    if (valid == 0) {
        // All values were NaN or Inf — nothing meaningful to compute
        stats.min_val  = 0.0f;
        stats.max_val  = 0.0f;
        stats.mean     = 0.0f;
        stats.std_dev  = 0.0f;
        stats.sparsity = 0.0f;
        return stats;
    }

    stats.min_val  = min_val;
    stats.max_val  = max_val;
    stats.mean     = static_cast<float>(sum / static_cast<double>(valid));
    stats.sparsity = static_cast<float>(zero_count) / static_cast<float>(valid);

   
    double mean_d  = static_cast<double>(stats.mean);
    double M2      = 0.0;
    size_t welford_n = 0;

    for (size_t idx : indices) {
        float v = data[idx];
        if (std::isnan(v) || std::isinf(v)) continue;
        ++welford_n;
        double delta = static_cast<double>(v) - mean_d;
        mean_d += delta / static_cast<double>(welford_n);
        double delta2 = static_cast<double>(v) - mean_d;
        M2 += delta * delta2;
    }

    stats.std_dev = (welford_n > 1)
        ? static_cast<float>(std::sqrt(M2 / static_cast<double>(welford_n - 1)))
        : 0.0f;

    return stats;
}





inline ActivationStats computeStats(const std::vector<float>& data,
                                    float threshold = kSparsityThreshold)
{
    return computeStats(data.data(), data.size(), threshold);
}



inline void softmax(float* data, size_t n) {
    if (!data || n == 0) return;

    float max_val = *std::max_element(data, data + n);
    float sum     = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        data[i]  = std::exp(data[i] - max_val);
        sum     += data[i];
    }

    if (sum > 0.0f) {
        for (size_t i = 0; i < n; ++i) data[i] /= sum;
    }
}




inline void normalizeToRange(float* data, size_t n) {
    if (!data || n == 0) return;

    float lo = *std::min_element(data, data + n);
    float hi = *std::max_element(data, data + n);
    float range = hi - lo;

    if (range < 1e-9f) {
        std::fill(data, data + n, 0.5f);
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        data[i] = (data[i] - lo) / range;
    }
}




inline void meanOverHeads(const float* src,
                          float*       dst,
                          int          num_heads,
                          int          seq_len)
{
    if (!src || !dst || num_heads <= 0 || seq_len <= 0) return;

    const size_t matrix_size = static_cast<size_t>(seq_len * seq_len);
    std::fill(dst, dst + matrix_size, 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        const float* head_ptr = src + h * matrix_size;
        for (size_t i = 0; i < matrix_size; ++i) {
            dst[i] += head_ptr[i];
        }
    }

    float inv_heads = 1.0f / static_cast<float>(num_heads);
    for (size_t i = 0; i < matrix_size; ++i) {
        dst[i] *= inv_heads;
    }
}




inline float clampToDisplay(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}




inline std::string formatLatency(int64_t micros) {
    if (micros < 0) return "---";

    char buf[32];
    if (micros < 1000) {
        std::snprintf(buf, sizeof(buf), "%lld µs", static_cast<long long>(micros));
    } else if (micros < 1'000'000) {
        std::snprintf(buf, sizeof(buf), "%.1f ms",
                      static_cast<double>(micros) / 1000.0);
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f s",
                      static_cast<double>(micros) / 1'000'000.0);
    }
    return std::string(buf);
}




inline float percentOf(int64_t part, int64_t total) {
    if (total <= 0) return 0.0f;
    return clampToDisplay(static_cast<float>(part) / static_cast<float>(total));
}

}
}