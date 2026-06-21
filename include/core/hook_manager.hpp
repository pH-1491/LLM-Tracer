#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <algorithm>

#include "core/layer_record.hpp"
#include "core/ring_buffer.hpp"
#include "instrumentation/metrics_collector.hpp"
#include "instrumentation/latency_tracker.hpp"

namespace llmtracer {
    class LatencyTracker;
    struct AttentionTensorView;
    struct TensorView;

    using HookHandle = uint64_t;
constexpr HookHandle kInvalidHookHandle = 0;

struct HookContext {

    std::string  layer_name;        // e.g. "transformer.h.3.attn"
    LayerType    layer_type;        // Enum tag for this sub-module
    int          layer_index;       // Position in the overall layer sequence
    uint64_t     pass_id;           // Monotonically increasing forward-pass ID


    TensorView   input;             // Tensor entering the layer
    TensorView   output;            // Tensor leaving the layer (empty in pre-hook)


    std::optional<AttentionTensorView> attention;


    TimePoint    hook_time;         // When this callback was invoked
};


using PreHookFn  = std::function<void(const HookContext&, LayerRecord&)>;
using PostHookFn = std::function<void(const HookContext&, LayerRecord&)>;



struct HookEntry {
    HookHandle   handle;
    std::string  layer_name;
    LayerType    layer_type;
    int          layer_index;
    PreHookFn    pre_fn;        // May be nullptr if only post-hook is needed
    PostHookFn   post_fn;       // May be nullptr if only pre-hook is needed
    bool         enabled;       // Hooks can be individually paused
};


class HookManager {
public:
    explicit HookManager(RingBuffer&          buffer,
                         LatencyTracker&      latency,
                         CollectionConfig     cfg = {})
        : buffer_(buffer)
        , latency_(latency)
        , collector_(std::move(cfg))
        , next_handle_(1)        // 0 is reserved for kInvalidHookHandle
        , next_record_id_(1)
        , next_pass_id_(1)
        , enabled_(true)
    {}

    ~HookManager() = default;

    // Non-copyable
    HookManager(const HookManager&)            = delete;
    HookManager& operator=(const HookManager&) = delete;


    HookHandle registerHook(const std::string& layer_name,
                             LayerType          layer_type,
                             int                layer_index,
                             PreHookFn          pre_fn,
                             PostHookFn         post_fn)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        HookHandle handle = next_handle_++;
        HookEntry entry {
            handle,
            layer_name,
            layer_type,
            layer_index,
            std::move(pre_fn),
            std::move(post_fn),
            true    // enabled by default
        };

        hooks_by_name_[layer_name].push_back(entries_.size());
        entries_.push_back(std::move(entry));

        return handle;
    }

    // Convenience: register only a post-hook (most common case — we want
    // output stats but don't need to intercept the input separately).
    HookHandle registerPostHook(const std::string& layer_name,
                                 LayerType          layer_type,
                                 int                layer_index,
                                 PostHookFn         post_fn)
    {
        return registerHook(layer_name, layer_type, layer_index,
                            nullptr, std::move(post_fn));
    }

    // Convenience: register a hook for a complete layer with automatic
    // metrics collection — no callback needed. The HookManager handles
    // stats, timing, and ring-buffer push internally.
    // This is the most ergonomic entry point for simple instrumentation.
    HookHandle registerAutoHook(const std::string& layer_name,
                                 LayerType          layer_type,
                                 int                layer_index)
    {
        return registerHook(layer_name, layer_type, layer_index,
                            nullptr, nullptr);
        // When both callbacks are nullptr, firePostHook still builds the
        // record from context data (shapes, timing, stats) automatically.
    }

    // Remove a previously registered hook.
    // Returns true if the handle was found and removed, false if not found.
    bool removeHook(HookHandle handle) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (auto& entry : entries_) {
            if (entry.handle == handle) {
                entry.enabled = false;  // Logical removal; slot stays for index stability
                // Also remove from name map
                auto& name_vec = hooks_by_name_[entry.layer_name];
                name_vec.erase(
                    std::remove_if(name_vec.begin(), name_vec.end(),
                        [&](size_t idx) { return entries_[idx].handle == handle; }),
                    name_vec.end()
                );
                return true;
            }
        }
        return false;
    }

    // Enable or disable a specific hook without removing it.
    bool setHookEnabled(HookHandle handle, bool enabled) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (auto& entry : entries_) {
            if (entry.handle == handle) {
                entry.enabled = enabled;
                return true;
            }
        }
        return false;
    }


    void setEnabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool isEnabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }


    uint64_t firePreHook(const std::string& layer_name,
                          const TensorView&  input,
                          std::optional<AttentionTensorView> attn = std::nullopt)
    {
        if (!enabled_.load(std::memory_order_relaxed)) return 0;

        // Look up registered hooks for this layer
        std::vector<size_t> indices = lookupIndices(layer_name);
        if (indices.empty()) return 0;

        uint64_t pass_id = next_pass_id_.fetch_add(1, std::memory_order_relaxed);

        // Build partial context (output not yet available)
        HookContext ctx;
        ctx.layer_name  = layer_name;
        ctx.pass_id     = pass_id;
        ctx.input       = input;
        ctx.attention   = attn;
        ctx.hook_time   = Clock::now();

        // Allocate a LayerRecord in the pending map, start timing
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);

            for (size_t idx : indices) {
                auto& entry = entries_[idx];
                if (!entry.enabled) continue;

                ctx.layer_type  = entry.layer_type;
                ctx.layer_index = entry.layer_index;

                // Start a new record
                uint64_t rec_id = next_record_id_.fetch_add(1, std::memory_order_relaxed);
                LayerRecord record(rec_id, layer_name, entry.layer_type, entry.layer_index);
                record.start_time = ctx.hook_time;

                // Collect input stats
                collector_.collectInputStats(input, record);

                // Fire user pre-hook callback if provided
                if (entry.pre_fn) {
                    try { entry.pre_fn(ctx, record); }
                    catch (...) { /* Never let hook callbacks crash the model */ }
                }

                pending_records_[pass_id] = std::move(record);
            }
        }

        return pass_id;
    }


    bool firePostHook(uint64_t           pass_id,
                      const TensorView&  output,
                      std::optional<AttentionTensorView> attn = std::nullopt)
    {
        if (!enabled_.load(std::memory_order_relaxed)) return false;
        if (pass_id == 0) return false;

        std::lock_guard<std::mutex> lock(pending_mutex_);

        auto it = pending_records_.find(pass_id);
        if (it == pending_records_.end()) return false;

        LayerRecord& record = it->second;

        // Stamp end time
        record.end_time = Clock::now();

        // Collect output stats
        collector_.collectOutputStats(output, record);

        // Collect attention data if available
        if (attn.has_value()) {
            collector_.collectAttention(attn.value(), record);
        }

        // Record latency in the tracker
        latency_.scopeTimerFor(record).stop();

        // Rebuild context for post-hook callbacks
        HookContext ctx;
        ctx.layer_name  = record.layer_name;
        ctx.layer_type  = record.layer_type;
        ctx.layer_index = record.layer_index;
        ctx.pass_id     = pass_id;
        ctx.output      = output;
        ctx.attention   = attn;
        ctx.hook_time   = record.end_time;

        // Fire post-hook callbacks
        {
            // Need registry lock to safely read entries_
            // Use try_lock to avoid deadlock if pre-hook is firing concurrently
            std::unique_lock<std::mutex> reg_lock(registry_mutex_, std::try_to_lock);
            if (reg_lock.owns_lock()) {
                auto indices = lookupIndicesUnlocked(record.layer_name);
                for (size_t idx : indices) {
                    auto& entry = entries_[idx];
                    if (!entry.enabled || !entry.post_fn) continue;
                    try { entry.post_fn(ctx, record); }
                    catch (...) { /* Swallow — never crash the model */ }
                }
            }
        }

        // Push completed record to ring buffer
        buffer_.push(std::move(record));

        // Clean up pending slot
        pending_records_.erase(it);

        return true;
    }


    std::vector<std::string> registeredLayerNames() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        std::vector<std::string> names;
        for (const auto& entry : entries_) {
            if (entry.enabled) names.push_back(entry.layer_name);
        }
        return names;
    }

    size_t hookCount() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        size_t n = 0;
        for (const auto& e : entries_) if (e.enabled) ++n;
        return n;
    }


    // Remove all registered hooks and clear pending records.
    void clear() {
        std::lock_guard<std::mutex> r(registry_mutex_);
        std::lock_guard<std::mutex> p(pending_mutex_);
        entries_.clear();
        hooks_by_name_.clear();
        pending_records_.clear();
    }

    // Start a new forward pass (increments pass_id counter).
    // Call this at the very beginning of each model forward() invocation
    // so timing spans align with actual pass boundaries.
    uint64_t beginPass() {
        return next_pass_id_.fetch_add(1, std::memory_order_relaxed);
    }

private:

    // Returns indices into entries_ for a given layer name.
    // Acquires registry_mutex_ internally.
    std::vector<size_t> lookupIndices(const std::string& name) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return lookupIndicesUnlocked(name);
    }

    // Same but caller must already hold registry_mutex_.
    std::vector<size_t> lookupIndicesUnlocked(const std::string& name) const {
        auto it = hooks_by_name_.find(name);
        if (it == hooks_by_name_.end()) return {};
        return it->second;
    }


    RingBuffer&       buffer_;
    LatencyTracker&   latency_;
    MetricsCollector  collector_;


    mutable std::mutex                              registry_mutex_;
    std::vector<HookEntry>                          entries_;
    std::unordered_map<std::string, std::vector<size_t>> hooks_by_name_;

    // ── Pending records (protected by pending_mutex_) ──────────────────────────
    // Records started in firePreHook() but not yet completed by firePostHook().
    // Key: pass_id.  Value: in-progress LayerRecord.
    std::mutex                                      pending_mutex_;
    std::unordered_map<uint64_t, LayerRecord>       pending_records_;

    // ── Atomic counters ────────────────────────────────────────────────────────
    std::atomic<HookHandle> next_handle_;
    std::atomic<uint64_t>   next_record_id_;
    std::atomic<uint64_t>   next_pass_id_;
    std::atomic<bool>       enabled_;
};


class ScopedLayerHook {
public:
    ScopedLayerHook(HookManager&       manager,
                    const std::string& layer_name,
                    const TensorView&  input,
                    std::optional<AttentionTensorView> attn = std::nullopt)
        : manager_(manager)
        , pass_id_(0)
        , output_set_(false)
    {
        pass_id_ = manager_.firePreHook(layer_name, input, attn);
        layer_name_ = layer_name;
    }

    // Call this after the layer's forward pass completes to provide output data.
    void setOutput(const TensorView& output,
                   std::optional<AttentionTensorView> attn = std::nullopt)
    {
        output_      = output;
        output_attn_ = attn;
        output_set_  = true;
    }

    ~ScopedLayerHook() {
        if (pass_id_ == 0) return;
        TensorView out = output_set_ ? output_ : TensorView{};
        manager_.firePostHook(pass_id_, out, output_attn_);
    }

    // Non-copyable
    ScopedLayerHook(const ScopedLayerHook&)            = delete;
    ScopedLayerHook& operator=(const ScopedLayerHook&) = delete;

private:
    HookManager&                        manager_;
    std::string                         layer_name_;
    uint64_t                            pass_id_;
    TensorView                          output_;
    std::optional<AttentionTensorView>  output_attn_;
    bool                                output_set_;
};

}  // namespace llmtracer