#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include <optional>
#include <algorithm>
#include <numeric>
#include <cstdint>

#include "core/layer_record.hpp"

namespace llmtracer {

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;
using Micros    = std::chrono::microseconds;


struct LayerTimingRecord {
    std::string layer_name;
    TimePoint   start;
    TimePoint   end;
    bool        complete = false;   // false if end() was never called

    int64_t durationMicros() const {
        if (!complete) return -1;
        return std::chrono::duration_cast<Micros>(end - start).count();
    }
};

struct LayerStats {
    std::string layer_name;

    uint64_t    call_count  = 0;        // How many times this layer was timed
    int64_t     total_us    = 0;        // Sum of all durations (microseconds)
    int64_t     min_us      = INT64_MAX;
    int64_t     max_us      = 0;
    int64_t     last_us     = 0;        // Most recent duration

    // Rolling window of the last N samples for a moving average

    static constexpr size_t kWindowSize = 32;
    std::vector<int64_t>    window;     // Circular; newest at back

    void record(int64_t duration_us) {
        if (duration_us < 0) return;    // Incomplete timer — skip
        ++call_count;
        total_us += duration_us;
        last_us   = duration_us;
        if (duration_us < min_us) min_us = duration_us;
        if (duration_us > max_us) max_us = duration_us;

        window.push_back(duration_us);
        if (window.size() > kWindowSize) {
            window.erase(window.begin());   // Keep window bounded
        }
    }

    // Average over all calls
    double meanUs() const {
        return (call_count == 0) ? 0.0
            : static_cast<double>(total_us) / static_cast<double>(call_count);
    }

    // Moving average over the last kWindowSize calls
    double movingAverageUs() const {
        if (window.empty()) return 0.0;
        int64_t sum = std::accumulate(window.begin(), window.end(), int64_t{0});
        return static_cast<double>(sum) / static_cast<double>(window.size());
    }

    // Standard deviation over the rolling window
    double stdDevUs() const {
        if (window.size() < 2) return 0.0;
        double mean = movingAverageUs();
        double sq_sum = 0.0;
        for (int64_t v : window) {
            double d = static_cast<double>(v) - mean;
            sq_sum += d * d;
        }
        return std::sqrt(sq_sum / static_cast<double>(window.size() - 1));
    }
};

class ScopeTimer {
public:
    using Callback = std::function<void(LayerTimingRecord)>;

    ScopeTimer(std::string layer_name, Callback on_done)
        : record_()
        , on_done_(std::move(on_done))
        , fired_(false)
    {
        record_.layer_name = std::move(layer_name);
        record_.start      = Clock::now();
        record_.complete   = false;
    }


    ScopeTimer(const ScopeTimer&)            = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;

    // Movable so it can be returned from factory functions
    ScopeTimer(ScopeTimer&& other) noexcept
        : record_(std::move(other.record_))
        , on_done_(std::move(other.on_done_))
        , fired_(other.fired_)
    {
        other.fired_ = true;    // Prevent double-fire on moved-from object
    }

    ~ScopeTimer() {
        if (!fired_) stop();
    }

    // Manually stop the timer before scope exit (optional — destructor handles it)
    void stop() {
        if (fired_) return;
        record_.end      = Clock::now();
        record_.complete = true;
        fired_           = true;
        if (on_done_) on_done_(std::move(record_));
    }

    // Read elapsed time so far without stopping the timer
    int64_t elapsedMicros() const {
        if (fired_) return record_.durationMicros();
        return std::chrono::duration_cast<Micros>(Clock::now() - record_.start).count();
    }

private:
    LayerTimingRecord record_;
    Callback          on_done_;
    bool              fired_;
};


class LatencyTracker {
public:
    LatencyTracker()  = default;
    ~LatencyTracker() = default;


    LatencyTracker(const LatencyTracker&)            = delete;
    LatencyTracker& operator=(const LatencyTracker&) = delete;

    [[nodiscard]]
    ScopeTimer scopeTimer(const std::string& layer_name) {
        return ScopeTimer(
            layer_name,
            [this](LayerTimingRecord r) { onTimerDone(std::move(r)); }
        );
    }


    [[nodiscard]]
    ScopeTimer scopeTimerFor(LayerRecord& target_record) {
        return ScopeTimer(
            target_record.layer_name,
            [this, &target_record](LayerTimingRecord r) {
                target_record.start_time = r.start;
                target_record.end_time   = r.end;
                onTimerDone(std::move(r));
            }
        );
    }


    std::vector<LayerStats> allStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<LayerStats> out;
        out.reserve(stats_.size());
        for (const auto& [name, s] : stats_) {
            out.push_back(s);
        }
        // Sort by layer name so TUI displays a stable order
        std::sort(out.begin(), out.end(),
                  [](const LayerStats& a, const LayerStats& b){
                      return a.layer_name < b.layer_name;
                  });
        return out;
    }

    // Get stats for a single named layer (returns nullopt if not seen yet)
    std::optional<LayerStats> statsFor(const std::string& layer_name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(layer_name);
        if (it == stats_.end()) return std::nullopt;
        return it->second;
    }

    // The highest `last_us` across all layers — used to normalize bar chart widths
    int64_t maxLastLatencyUs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t max_val = 0;
        for (const auto& [_, s] : stats_) {
            if (s.last_us > max_val) max_val = s.last_us;
        }
        return max_val;
    }

    // The highest moving-average latency — used for a smoothed bar chart view
    double maxMovingAverageUs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        double max_val = 0.0;
        for (const auto& [_, s] : stats_) {
            double ma = s.movingAverageUs();
            if (ma > max_val) max_val = ma;
        }
        return max_val;
    }

    // Total time (sum of last_us for all layers) — approximate forward-pass cost
    int64_t totalLastPassUs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t total = 0;
        for (const auto& [_, s] : stats_) {
            total += s.last_us;
        }
        return total;
    }

    // Number of unique layers seen so far
    size_t layerCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_.size();
    }



    // Wipe all accumulated stats
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
    }

private:
    // Called from ScopeTimer destructor (instrumentation thread)
    void onTimerDone(LayerTimingRecord record) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = stats_[record.layer_name];   // insert-or-lookup
        if (s.layer_name.empty()) {
            s.layer_name = record.layer_name;
        }
        s.record(record.durationMicros());
    }

    mutable std::mutex                            mutex_;
    std::unordered_map<std::string, LayerStats>   stats_;
};


class GlobalLatencyTracker {
public:
    static LatencyTracker& instance() {
        static LatencyTracker tracker;
        return tracker;
    }
private:
    GlobalLatencyTracker() = delete;
};

}