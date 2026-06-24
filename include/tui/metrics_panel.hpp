#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "panel_base.hpp"
#include "tui/panel_base.hpp"
#include "core/layer_record.hpp"
#include "utils/math_utils.hpp"

namespace llmtracer {
namespace tui {


class MetricsPanel : public PanelBase {
public:
    MetricsPanel() : PanelBase(PanelId::Metrics) {}


    void init(const Rect& rect) override {
        PanelBase::init(rect);
    }

    void resize(const Rect& rect) override {
        PanelBase::resize(rect);
    }

    void render(const RenderState& state) override {
        if (!win_) return;

        drawBorder(" Metrics ");

        if (!state.selected_record.has_value()) {
            writeAt(1, 1, "Select a layer in the Layers",
                    COLOR_PAIR(colorPairMuted()));
            writeAt(2, 1, "panel to view its metrics.",
                    COLOR_PAIR(colorPairMuted()));
            refresh();
            return;
        }

        const LayerRecord& rec = state.selected_record.value();

        int row = 0;
        row = drawIdentity(rec, row);
        row = drawDivider(row, "Tensor Shapes");
        row = drawShapes(rec, row);
        row = drawDivider(row, "Input Activations");
        row = drawStats(rec.input_stats,  row);
        row = drawDivider(row, "Output Activations");
        row = drawStats(rec.output_stats, row);
        row = drawDivider(row, "Diagnostics");
        row = drawDiagnostics(rec, row);

        refresh();
    }

    bool onKey(const KeyEvent& ) override {
        // Metrics panel is display-only — no interactive controls
        return false;
    }

    std::string statusHint() const override {
        return "Metrics  (display only)  |  Tab: switch panel  |  q: quit";
    }

private:

    int drawIdentity(const LayerRecord& rec, int row) {
        int w = rect_.innerWidth();
        if (w < 10 || row >= rect_.innerHeight()) return row;


        {
            std::string label = "Name: ";
            std::string name  = rec.layer_name;
            int avail = w - (int)label.size() - 2;
            if ((int)name.size() > avail && avail > 3)
                name = "…" + name.substr(name.size() - (avail - 1));
            writeAt(row, 1, label,  COLOR_PAIR(colorPairMuted()));
            writeAt(row, 1 + (int)label.size(), name, A_BOLD);
            ++row;
        }

        // Type + index + record ID
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "Type: %-8s  Idx: %-3d  #%llu",
                          layerTypeToString(rec.layer_type),
                          rec.layer_index,
                          static_cast<unsigned long long>(rec.record_id));
            int badge_pair = layerBadgePair(rec.layer_type);
            // Write "Type: " in muted, then badge value in colour, rest normal
            writeAt(row, 1, std::string(buf), COLOR_PAIR(badge_pair));
            ++row;
        }

        // Latency
        {
            int64_t us = rec.latencyMicros();
            std::string lat = math::formatLatency(us);
            char buf[48];
            std::snprintf(buf, sizeof(buf), "Time: %s", lat.c_str());
            int lat_pair = (us > 10000) ? colorPairWarn() : colorPairOk();
            writeAt(row, 1, std::string(buf), COLOR_PAIR(lat_pair) | A_BOLD);
            ++row;
        }

        return row + 1;  // blank line after section
    }

    // Tensor shapes block
    int drawShapes(const LayerRecord& rec, int row) {
        int w = rect_.innerWidth();
        if (w < 10 || row >= rect_.innerHeight()) return row;

        auto drawShapeLine = [&](const char* label, const TensorShape& shape) {
            if (row >= rect_.innerHeight()) return;
            char buf[64];
            std::string shape_str = shape.dims.empty() ? "N/A" : shape.toString();
            int64_t n = shape.numel();
            if (n > 0) {
                std::snprintf(buf, sizeof(buf), "%-8s %-20s  n=%lld",
                              label,
                              shape_str.c_str(),
                              static_cast<long long>(n));
            } else {
                std::snprintf(buf, sizeof(buf), "%-8s %s", label, shape_str.c_str());
            }
            writeAt(row, 1, std::string(buf));
            ++row;
        };

        drawShapeLine("Input: ", rec.input_shape);
        drawShapeLine("Output:", rec.output_shape);

        return row + 1;
    }

    // Activation statistics block (used for both input and output stats)
    int drawStats(const ActivationStats& s, int row) {
        int w = rect_.innerWidth() - 2;
        if (w < 20 || row >= rect_.innerHeight()) return row;

        // ── Column headers ───────────────────────────────────────────────────
        if (row < rect_.innerHeight()) {
            writeAt(row, 1, "Mean     Std      Min      Max      Sparse",
                    COLOR_PAIR(colorPairMuted()));
            ++row;
        }


        if (row < rect_.innerHeight()) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "%-8.4f %-8.4f %-8.4f %-8.4f %.1f%%",
                          static_cast<double>(s.mean),
                          static_cast<double>(s.std_dev),
                          static_cast<double>(s.min_val),
                          static_cast<double>(s.max_val),
                          static_cast<double>(s.sparsity * 100.0f));

            int val_pair = statColorPair(s);
            writeAt(row, 1, std::string(buf), COLOR_PAIR(val_pair));
            ++row;
        }


        if ((s.has_nan || s.has_inf) && row < rect_.innerHeight()) {
            std::string flags;
            if (s.has_nan) flags += "  [NaN] ";
            if (s.has_inf) flags += "  [Inf] ";
            writeAt(row, 1, flags, COLOR_PAIR(colorPairError()) | A_BOLD | A_BLINK);
            ++row;
        }


        if (row < rect_.innerHeight()) {
            int bar_w = std::min(w - 6, 32);  // leave room for " XX%"
            if (bar_w > 4) {
                drawHealthBar(row, s.sparsity, bar_w);
                ++row;
            }
        }

        return row + 1;  // blank line between sections
    }


    void drawHealthBar(int row, float sparsity, int bar_width) {
        float density   = 1.0f - sparsity;
        int   filled    = static_cast<int>(density * bar_width);
        int   empty     = bar_width - filled;

        std::string bar = "[";

        // Filled portion
        for (int i = 0; i < filled; ++i) bar += "█";

        // Empty portion
        for (int i = 0; i < empty;  ++i) bar += "░";

        bar += "]";

        // Append density percentage
        char pct[12];
        std::snprintf(pct, sizeof(pct), " %3.0f%% dense",
                      static_cast<double>(density * 100.0f));
        bar += pct;

        // Colour: green when dense, yellow when getting sparse, red when very sparse
        int pair;
        if (sparsity < 0.3f)      pair = colorPairOk();
        else if (sparsity < 0.7f) pair = colorPairWarn();
        else                       pair = colorPairError();

        writeAt(row, 1, bar, COLOR_PAIR(pair));
    }

    // Diagnostics block — plain-english anomaly summary
    int drawDiagnostics(const LayerRecord& rec, int row) {
        if (row >= rect_.innerHeight()) return row;

        // Collect all anomalies
        std::vector<std::string> issues;

        auto check = [&](const ActivationStats& s, const char* side) {
            if (s.has_nan) {
                issues.push_back(std::string("⚡ ") + side + ": NaN values detected");
            }
            if (s.has_inf) {
                issues.push_back(std::string("⚡ ") + side + ": Inf values detected");
            }
            if (s.sparsity > 0.9f) {
                char buf[64];
                std::snprintf(buf, sizeof(buf),
                              "⚠  %s: %.0f%% sparse — dead neurons?",
                              side, static_cast<double>(s.sparsity * 100.0f));
                issues.push_back(std::string(buf));
            }
            if (std::abs(s.max_val) > 1e4f) {
                char buf[64];
                std::snprintf(buf, sizeof(buf),
                              "⚠  %s: max=%.2e — clipping risk",
                              side, static_cast<double>(s.max_val));
                issues.push_back(std::string(buf));
            }
            if (std::abs(s.min_val) > 1e4f) {
                char buf[64];
                std::snprintf(buf, sizeof(buf),
                              "⚠  %s: min=%.2e — clipping risk",
                              side, static_cast<double>(s.min_val));
                issues.push_back(std::string(buf));
            }
        };

        check(rec.input_stats,  "input");
        check(rec.output_stats, "output");

        if (issues.empty()) {
            writeAt(row, 1, "✓ No anomalies detected",
                    COLOR_PAIR(colorPairOk()) | A_BOLD);
            ++row;
        } else {
            for (const auto& msg : issues) {
                if (row >= rect_.innerHeight()) break;
                int pair = (msg[0] == '⚡') ? colorPairError() : colorPairWarn();
                writeAt(row, 1, msg, COLOR_PAIR(pair) | A_BOLD);
                ++row;
            }
        }

        return row;
    }


    int drawDivider(int row, const char* label) {
        if (row >= rect_.innerHeight()) return row;

        int w     = rect_.innerWidth() - 2;
        std::string line = "── ";
        line += label;
        line += " ";
        while ((int)line.size() < w) line += "─";
        line = line.substr(0, w);

        writeAt(row, 1, line, COLOR_PAIR(colorPairMuted()));
        return row + 1;
    }

   

    static int statColorPair(const ActivationStats& s) {
        if (s.has_nan || s.has_inf)  return colorPairError();
        if (s.sparsity > 0.9f)       return colorPairWarn();
        if (s.sparsity > 0.5f)       return colorPairWarn();
        return colorPairOk();
    }

    static int layerBadgePair(LayerType t) {
        switch (t) {
            case LayerType::Embedding:  return colorPairEmbed();
            case LayerType::Attention:  return colorPairAttn();
            case LayerType::MLP:        return colorPairMlp();
            case LayerType::LayerNorm:  return colorPairNorm();
            case LayerType::Output:     return colorPairOut();
            default:                    return colorPairMuted();
        }
    }
};

} // namespace tui
} // namespace llmtracer