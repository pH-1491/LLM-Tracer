#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <functional>
#include <stdexcept>

#include "core/layer_record.hpp"

namespace llmtracer {


class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity)
        , head_(0)
        , tail_(0)
        , count_(0)
        , total_written_(0)
    {
        if (capacity == 0) {
            throw std::invalid_argument("RingBuffer: capacity must be > 0");
        }
        storage_.resize(capacity);
    }

    // Non-copyable, non-movable (holds a mutex)
    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&)                 = delete;
    RingBuffer& operator=(RingBuffer&&)      = delete;

    ~RingBuffer() = default;


    // Push a LayerRecord by copy.
    // If the buffer is full, overwrites the oldest record (head advances).
    void push(const LayerRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        pushImpl(record);
    }

    // Push a LayerRecord by move (preferred — avoids copying AttentionData.matrix)
    void push(LayerRecord&& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        pushImpl(std::move(record));
    }

    // Emplace a LayerRecord in-place (construct directly into the slot)
    template<typename... Args>
    void emplace(Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        storage_[tail_] = LayerRecord(std::forward<Args>(args)...);
        advanceTail();
    }



    // Returns the number of valid records currently stored.
    size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    // Returns the maximum number of records the buffer can hold.
    size_t capacity() const noexcept { return capacity_; }

    // Returns true if the buffer has no valid records.
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ == 0;
    }

    // Returns true if the buffer is at full capacity (next push will overwrite).
    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ == capacity_;
    }

    uint64_t totalWritten() const {
        return total_written_.load(std::memory_order_relaxed);
    }



    std::optional<LayerRecord> at(size_t idx) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (idx >= count_) return std::nullopt;

        size_t physical = (head_ + idx) % capacity_;
        return storage_[physical];   // copy
    }

    // Read the most recently written record.
    std::optional<LayerRecord> latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) return std::nullopt;

        size_t physical = (tail_ + capacity_ - 1) % capacity_;
        return storage_[physical];   // copy
    }

    // Read the oldest record still in the buffer.
    std::optional<LayerRecord> oldest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) return std::nullopt;
        return storage_[head_];      // copy
    }



    void snapshot(std::vector<LayerRecord>& out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        out.clear();
        out.reserve(count_);
        for (size_t i = 0; i < count_; ++i) {
            size_t physical = (head_ + i) % capacity_;
            out.push_back(storage_[physical]);
        }
    }


    void snapshotWhere(std::vector<LayerRecord>& out,
                       std::function<bool(const LayerRecord&)> pred) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        out.clear();
        for (size_t i = 0; i < count_; ++i) {
            size_t physical = (head_ + i) % capacity_;
            const auto& rec = storage_[physical];
            if (pred(rec)) out.push_back(rec);
        }
    }




    void forEach(std::function<void(const LayerRecord&, size_t)> fn) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count_; ++i) {
            size_t physical = (head_ + i) % capacity_;
            fn(storage_[physical], i);
        }
    }



    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_  = 0;
        tail_  = 0;
        count_ = 0;

    }

    void resize(size_t new_capacity) {
        if (new_capacity == 0) {
            throw std::invalid_argument("RingBuffer: capacity must be > 0");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        storage_.clear();
        storage_.resize(new_capacity);
        capacity_ = new_capacity;
        head_     = 0;
        tail_     = 0;
        count_    = 0;
    }



    // Returns fill percentage
    float fillRatio() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<float>(count_) / static_cast<float>(capacity_);
    }

    // Returns how many records were overwritten (lost) due to buffer being full.
    uint64_t overwriteCount() const {
        return overwrite_count_.load(std::memory_order_relaxed);
    }

    std::string statusLine() const {
        std::lock_guard<std::mutex> lock(mutex_);
        char buf[64];
        std::snprintf(buf, sizeof(buf),
                      "Buffer: %zu / %zu records  [%llu written, %llu overwritten]",
                      count_, capacity_,
                      static_cast<unsigned long long>(total_written_.load()),
                      static_cast<unsigned long long>(overwrite_count_.load()));
        return std::string(buf);
    }

private:

    // Must be called with mutex_ held.
    void advanceTail() {
        tail_ = (tail_ + 1) % capacity_;
        if (count_ < capacity_) {
            ++count_;
        } else {
            // Buffer was full — the oldest slot (head_) was just overwritten.
            head_ = (head_ + 1) % capacity_;
            overwrite_count_.fetch_add(1, std::memory_order_relaxed);
        }
        total_written_.fetch_add(1, std::memory_order_relaxed);
    }

    template<typename U>
    void pushImpl(U&& record) {
        storage_[tail_] = std::forward<U>(record);
        advanceTail();
    }



    size_t                  capacity_;       // Max records (fixed after resize)
    size_t                  head_;           // Index of oldest valid record
    size_t                  tail_;           // Index where next write goes
    size_t                  count_;          // Current number of valid records
    std::vector<LayerRecord> storage_;       // Pre-allocated backing store
    mutable std::mutex      mutex_;          // Guards head_, tail_, count_, storage_

    // Atomic counters readable without holding the mutex (for status display)
    std::atomic<uint64_t>   total_written_{0};
    std::atomic<uint64_t>   overwrite_count_{0};
};




class RingBufferView {
public:
    explicit RingBufferView(const RingBuffer& buf) : buf_(buf) {}

    size_t   count()        const { return buf_.count(); }
    size_t   capacity()     const { return buf_.capacity(); }
    bool     empty()        const { return buf_.empty(); }
    uint64_t totalWritten() const { return buf_.totalWritten(); }
    float    fillRatio()    const { return buf_.fillRatio(); }
    uint64_t overwriteCount() const { return buf_.overwriteCount(); }
    std::string statusLine()  const { return buf_.statusLine(); }

    std::optional<LayerRecord> at(size_t idx)  const { return buf_.at(idx); }
    std::optional<LayerRecord> latest()         const { return buf_.latest(); }
    std::optional<LayerRecord> oldest()         const { return buf_.oldest(); }

    void snapshot(std::vector<LayerRecord>& out) const {
        buf_.snapshot(out);
    }

    void snapshotWhere(std::vector<LayerRecord>& out,
                       std::function<bool(const LayerRecord&)> pred) const {
        buf_.snapshotWhere(out, std::move(pred));
    }

    void forEach(std::function<void(const LayerRecord&, size_t)> fn) const {
        buf_.forEach(std::move(fn));
    }

private:
    const RingBuffer& buf_;
};

}