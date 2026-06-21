#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <optional>
#include <cstdint>
#include <cstring>

#include "metrics_collector.hpp"
#include "core/hook_manager.hpp"
#include "core/layer_record.hpp"
#include "instrumentation/metrics_collector.hpp"

namespace llmtracer {
    class HookManager;


    using ForwardFn = std::function<
    std::vector<float>(const float* input, const std::vector<int64_t>& shape)
>;

struct AttentionForwardResult {
    std::vector<float>   output;         // [batch, seq, dim] activations
    std::vector<float>   attn_weights;   // [batch, heads, seq, seq] softmax weights
    std::vector<int64_t> output_shape;
    int                  num_heads = 0;
    int                  seq_len   = 0;
};

using AttentionForwardFn = std::function<
    AttentionForwardResult(const float*              input,
                           const std::vector<int64_t>& shape)
>;

struct LayerShimConfig {
    std::string  layer_name;
    LayerType    layer_type   = LayerType::Unknown;
    int          layer_index  = -1;
    bool         capture_attn = false;   // Only meaningful for Attention layers
};


class ForwardHook {
public:
    ForwardHook(HookManager&         manager,
                LayerShimConfig      cfg)
        : manager_(manager)
        , cfg_(std::move(cfg))
    {}



    void setForwardFn(ForwardFn fn) {
        forward_fn_ = std::move(fn);
    }

    void setAttentionForwardFn(AttentionForwardFn fn) {
        attn_forward_fn_ = std::move(fn);
    }


    std::vector<float> operator()(const float*              input_ptr,
                                   const std::vector<int64_t>& input_shape)
    {
        if (!forward_fn_) {
            throw std::runtime_error(
                "ForwardHook: no forward function set for layer '" +
                cfg_.layer_name + "'");
        }


        TensorView input_view;
        input_view.data  = input_ptr;
        input_view.shape = input_shape;
        input_view.name  = cfg_.layer_name + ":input";


        uint64_t pass_id = manager_.firePreHook(cfg_.layer_name, input_view);


        std::vector<float> output;
        try {
            output = forward_fn_(input_ptr, input_shape);
        } catch (...) {
            // Forward pass threw — fire post-hook with empty output so the
            // record is completed (not leaked in pending_records_), then rethrow
            TensorView empty{};
            manager_.firePostHook(pass_id, empty);
            throw;
        }


        TensorView output_view;
        output_view.data  = output.data();
        output_view.shape = inferOutputShape(input_shape, output.size());
        output_view.name  = cfg_.layer_name + ":output";


        manager_.firePostHook(pass_id, output_view);

        return output;
    }

    // Attention layer invocation — captures attention weights alongside output.
    AttentionForwardResult callAttention(const float*              input_ptr,
                                          const std::vector<int64_t>& input_shape)
    {
        if (!attn_forward_fn_) {
            throw std::runtime_error(
                "ForwardHook: no attention forward function set for layer '" +
                cfg_.layer_name + "'");
        }


        TensorView input_view;
        input_view.data  = input_ptr;
        input_view.shape = input_shape;
        input_view.name  = cfg_.layer_name + ":input";


        uint64_t pass_id = manager_.firePreHook(cfg_.layer_name, input_view);


        AttentionForwardResult result;
        try {
            result = attn_forward_fn_(input_ptr, input_shape);
        } catch (...) {
            TensorView empty{};
            manager_.firePostHook(pass_id, empty);
            throw;
        }


        TensorView output_view;
        output_view.data  = result.output.data();
        output_view.shape = result.output_shape.empty()
            ? inferOutputShape(input_shape, result.output.size())
            : result.output_shape;
        output_view.name  = cfg_.layer_name + ":output";


        std::optional<AttentionTensorView> attn_view;
        if (!result.attn_weights.empty() && result.num_heads > 0) {
            AttentionTensorView av;
            av.data      = result.attn_weights.data();
            av.num_heads = result.num_heads;
            av.seq_len   = result.seq_len;
            av.batch     = 0;
            attn_view    = av;
        }


        manager_.firePostHook(pass_id, output_view, attn_view);

        return result;
    }


    const LayerShimConfig& config() const { return cfg_; }
    const std::string&     layerName() const { return cfg_.layer_name; }

private:
    // Infer output shape from the flat output size + input shape.
    // Most transformer layers preserve [batch, seq, dim] — only dim may change.
    // This is a best-effort inference; real shape should come from the model.
    static std::vector<int64_t> inferOutputShape(const std::vector<int64_t>& input_shape,
                                                   size_t                       output_numel)
    {
        if (input_shape.empty()) return { static_cast<int64_t>(output_numel) };

        // Assume batch and seq dims are preserved, only the last dim changes
        if (input_shape.size() >= 2) {
            int64_t batch_seq = 1;
            for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
                batch_seq *= input_shape[i];
            }
            int64_t out_dim = (batch_seq > 0)
                ? static_cast<int64_t>(output_numel) / batch_seq
                : static_cast<int64_t>(output_numel);

            std::vector<int64_t> shape(input_shape.begin(),
                                        input_shape.begin() + input_shape.size() - 1);
            shape.push_back(out_dim);
            return shape;
        }

        return { static_cast<int64_t>(output_numel) };
    }

    HookManager&       manager_;
    LayerShimConfig    cfg_;
    ForwardFn          forward_fn_;
    AttentionForwardFn attn_forward_fn_;
};


class ForwardHookRegistry {
public:
    explicit ForwardHookRegistry(HookManager& manager)
        : manager_(manager)
        , layer_counter_(0)
    {}


    ForwardHook* registerLayer(const std::string& name,
                                LayerType          type,
                                ForwardFn          fn)
    {
        int idx = layer_counter_++;
        LayerShimConfig cfg { name, type, idx, false };

        manager_.registerAutoHook(name, type, idx);

        auto hook = std::make_unique<ForwardHook>(manager_, cfg);
        hook->setForwardFn(std::move(fn));

        ForwardHook* ptr = hook.get();
        std::lock_guard<std::mutex> lock(mutex_);
        hooks_[name] = std::move(hook);
        return ptr;
    }


    ForwardHook* registerAttentionLayer(const std::string& name,
                                         int                num_heads,
                                         AttentionForwardFn fn)
    {
        int idx = layer_counter_++;
        LayerShimConfig cfg { name, LayerType::Attention, idx, true };
        cfg.capture_attn = true;

        manager_.registerAutoHook(name, LayerType::Attention, idx);

        auto hook = std::make_unique<ForwardHook>(manager_, cfg);
        hook->setAttentionForwardFn(std::move(fn));

        ForwardHook* ptr = hook.get();
        std::lock_guard<std::mutex> lock(mutex_);
        hooks_[name] = std::move(hook);
        return ptr;
    }


    struct TransformerBlockFns {
        ForwardFn          pre_norm_fn;     // LayerNorm before attention
        AttentionForwardFn attention_fn;    // Multi-head self-attention
        ForwardFn          post_norm_fn;    // LayerNorm before MLP
        ForwardFn          mlp_fn;          // Feed-forward network
        int                num_heads = 0;   // Attention head count
    };

    void registerTransformerBlock(int                    block_index,
                                   TransformerBlockFns    fns)
    {
        auto prefix = "transformer.h." + std::to_string(block_index);

        if (fns.pre_norm_fn) {
            registerLayer(prefix + ".ln_1", LayerType::LayerNorm,
                          std::move(fns.pre_norm_fn));
        }
        if (fns.attention_fn) {
            registerAttentionLayer(prefix + ".attn", fns.num_heads,
                                   std::move(fns.attention_fn));
        }
        if (fns.post_norm_fn) {
            registerLayer(prefix + ".ln_2", LayerType::LayerNorm,
                          std::move(fns.post_norm_fn));
        }
        if (fns.mlp_fn) {
            registerLayer(prefix + ".mlp", LayerType::MLP,
                          std::move(fns.mlp_fn));
        }
    }

    // Register the embedding layer (first layer, before transformer blocks)
    ForwardHook* registerEmbedding(const std::string& name, ForwardFn fn) {
        return registerLayer(name, LayerType::Embedding, std::move(fn));
    }

    // Register the output projection (last layer, after all transformer blocks)
    ForwardHook* registerOutput(const std::string& name, ForwardFn fn) {
        return registerLayer(name, LayerType::Output, std::move(fn));
    }



    // Get a hook by layer name. Returns nullptr if not found.
    ForwardHook* getHook(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = hooks_.find(name);
        return (it != hooks_.end()) ? it->second.get() : nullptr;
    }

    size_t hookCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return hooks_.size();
    }



    // Remove all registered shims
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        hooks_.clear();
        layer_counter_ = 0;
    }

private:
    HookManager& manager_;
    int          layer_counter_;    // Auto-incrementing layer_index

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ForwardHook>> hooks_;
};


inline std::unique_ptr<ForwardHook>
makePassthroughHook(HookManager&       manager,
                    LayerShimConfig    cfg,
                    ForwardFn          fn)
{
    manager.registerAutoHook(cfg.layer_name, cfg.layer_type, cfg.layer_index);
    auto hook = std::make_unique<ForwardHook>(manager, std::move(cfg));
    hook->setForwardFn(std::move(fn));
    return hook;
}

}  // namespace llmtracer