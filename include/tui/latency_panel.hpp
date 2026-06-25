#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>

#include "panel_base.hpp"
#include "tui/panel_base.hpp"
#include "core/layer_record.hpp"
#include "core/ring_buffer.hpp"
#include "instrumentation/latency_tracker.hpp"
#include "utils/math_utils.hpp"

namespace llmtracer {
namespace tui {

enum class LatencySortMode {
    ByLatency,      // Slowest first (default)
    ByLayerOrder,   // Layer index ascending (model order)
    ByName          // Alphabetical by layer name
};

inline const char* sortModeName(LatencySortMode m) {
    switch (m) {
        case LatencySortMode::ByLatency:    return "by-latency";
        case LatencySortMode::ByLayerOrder: return "by-order";
        case LatencySortMode::ByName:       return "by-name";
        default:                             return "unknown";
    }
}

class LatencyPanel : public PanelBase {
public:
    explicit LatencyPanel(const LatencyTracker& tracker)
        : PanelBase(PanelId::Latency)
        , tracker_(tracker)
        , sort_mode_(LatencySortMode::ByLatency)
        , use_moving_avg_(false)
        , scroll_top_(0)
    {}



    void init(const Rect& rect) override {
        PanelBase::init(rect);
    }

    void resize(const Rect& rect) override {
        PanelBase::resize(rect);
        clampScroll();
    }

    void render(const RenderState& state) override {
        if (!win_) return;

        // Pull latest stats from tracker every frame
        stats_ = tracker_.allStats();

        if (stats_.empty()) {
            drawBorder(" Latency ");
            writeAt(1, 1, "No timing data yet.",
                    COLOR_PAIR(colorPairMuted()));
            writeAt(2, 1, "Waiting for first forward pass...",
                    COLOR_PAIR(colorPairMuted()));
            refresh();
            return;
        }

        sortStats();

        // Compute max latency for bar normalisation
        double max_us = computeMaxUs();

        // Total forward pass cost (sum of all last-pass latencies)
        int64_t total_us = tracker_.totalLastPassUs();

        drawBorder(buildTitle(total_us, state));
        drawBars(max_us);
        drawFooter();

        refresh();
    }

    bool onKey(const KeyEvent& key) override {
        if (key.isDown()) {
            ++scroll_top_;
            clampScroll();
            return true;
        }
        if (key.isUp()) {
            if (scroll_top_ > 0) --scroll_top_;
            return true;
        }
        if (key.isHome()) {
            scroll_top_ = 0;
            return true;
        }
        if (key.isEnd()) {
            if (!stats_.empty()) {
                int visible = std::max(1, rect_.innerHeight() - 2);
                scroll_top_ = (stats_.size() > (size_t)visible)
                    ? stats_.size() - visible : 0;
            }
            return true;
        }
        if (key.isPageDown()) {
            int page = std::max(1, rect_.innerHeight() - 2);
            scroll_top_ += page;
            clampScroll();
            return true;
        }
        if (key.isPageUp()) {
            int page = std::max(1, rect_.innerHeight() - 2);
            scroll_top_ = (scroll_top_ >= (size_t)page)
                ? scroll_top_ - page : 0;
            return true;
        }
        if (key.isToggle()) {
            // Cycle sort mode
            switch (sort_mode_) {
                case LatencySortMode::ByLatency:
                    sort_mode_ = LatencySortMode::ByLayerOrder; break;
                case LatencySortMode::ByLayerOrder:
                    sort_mode_ = LatencySortMode::ByName;       break;
                case LatencySortMode::ByName:
                    sort_mode_ = LatencySortMode::ByLatency;    break;
            }
            scroll_top_ = 0;
            return true;
        }
        if (key.raw == 'm' || key.raw == 'M') {
            use_moving_avg_ = !use_moving_avg_;
            return true;
        }
        return false;
    }

    std::string statusHint() const override {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "Latency  j/k:scroll  t:sort[%s]  m:avg[%s]  Tab:switch",
            sortModeName(sort_mode_),
            use_moving_avg_ ? "moving" : "last");
        return std::string(buf);
    }

private:


    std::string buildTitle(int64_t total_us, const RenderState& state) const {
        char buf[80];
        std::string total_str = math::formatLatency(total_us);
        std::snprintf(buf, sizeof(buf),
                      " Latency  total:%s  [%s%s] ",
                      total_str.c_str(),
                      sortModeName(sort_mode_),
                      use_moving_avg_ ? ",avg" : "");
        if (state.paused) return std::string(buf) + "⏸ ";
        return std::string(buf);
    }



    void drawBars(double max_us) {
        int visible_rows = rect_.innerHeight() - 2;  // reserve footer
        if (visible_rows <= 0) return;
        if (max_us <= 0.0) max_us = 1.0;  // guard against div-by-zero

        int inner_w = rect_.innerWidth() - 2;
        if (inner_w < 20) return;

        // Layout per row:
        //   [name: N chars][space][bar: B chars][space][latency: 8 chars]
        // We give name 35% of inner width, latency 8 chars, bar gets the rest
        const int lat_w  = 8;
        const int name_w = std::max(8, static_cast<int>(inner_w * 0.35));
        const int bar_w  = std::max(4, inner_w - name_w - lat_w - 2);

        for (int row = 0; row < visible_rows; ++row) {
            size_t stat_idx = scroll_top_ + static_cast<size_t>(row);
            if (stat_idx >= stats_.size()) {
                clearRow(row);
                continue;
            }

            const LayerStats& s = stats_[stat_idx];
            drawBarRow(row, s, name_w, bar_w, lat_w, max_us);
        }
    }

    void drawBarRow(int             row,
                    const LayerStats& s,
                    int             name_w,
                    int             bar_w,
                    int             lat_w,
                    double          max_us)
    {

        double us_val = use_moving_avg_
            ? s.movingAverageUs()
            : static_cast<double>(s.last_us);

        float fraction = math::percentOf(
            static_cast<int64_t>(us_val),
            static_cast<int64_t>(max_us));


        std::string name = s.layer_name;
        if ((int)name.size() > name_w) {
            name = "…" + name.substr(name.size() - (name_w - 1));
        }
        while ((int)name.size() < name_w) name += ' ';

        int filled_cells = static_cast<int>(fraction * bar_w);
        // Fractional remainder for the partial end block
        float remainder  = (fraction * bar_w) - filled_cells;

        int bar_color_pair = barColorPair(fraction);


        std::string lat_str = math::formatLatency(static_cast<int64_t>(us_val));
        // Right-align to lat_w chars
        while ((int)lat_str.size() < lat_w) lat_str = " " + lat_str;
        if ((int)lat_str.size() > lat_w) lat_str = lat_str.substr(0, lat_w);


        int col = 0;

        // Name
        writeAt(row, col, name, COLOR_PAIR(colorPairMuted()));
        col += name_w;

        // Space
        writeAt(row, col, " ");
        col += 1;


        {
            wattron(win_, COLOR_PAIR(bar_color_pair) | A_BOLD);
            int abs_row = rect_.innerRow() + row;
            int abs_col = rect_.innerCol() + col;
            for (int b = 0; b < filled_cells && b < bar_w; ++b) {
                mvwaddstr(win_, abs_row, abs_col + b, "█");
            }
            // Fractional end block
            if (filled_cells < bar_w) {
                const char* frac_char = math::fractionalBlock(remainder);
                mvwaddstr(win_, abs_row, abs_col + filled_cells, frac_char);
            }
            wattroff(win_, COLOR_PAIR(bar_color_pair) | A_BOLD);
        }

        // Bar — empty (background) portion
        {
            wattron(win_, COLOR_PAIR(colorPairMuted()));
            int abs_row = rect_.innerRow() + row;
            int abs_col = rect_.innerCol() + col + filled_cells + 1;
            int empty   = bar_w - filled_cells - 1;
            for (int b = 0; b < empty; ++b) {
                mvwaddstr(win_, abs_row, abs_col + b, "░");
            }
            wattroff(win_, COLOR_PAIR(colorPairMuted()));
        }

        col += bar_w;

        // Space
        writeAt(row, col, " ");
        col += 1;

        // Latency value
        writeAt(row, col, lat_str,
                COLOR_PAIR(bar_color_pair) | A_BOLD);
    }



    void drawFooter() {
        int footer_row = rect_.innerHeight() - 1;
        if (footer_row < 0) return;

        char buf[80];
        std::snprintf(buf, sizeof(buf),
                      " j/k:scroll  t:sort  m:avg  [%zu layers]",
                      stats_.size());

        writeAt(footer_row, 0, std::string(buf),
                COLOR_PAIR(colorPairMuted()));
    }



    void sortStats() {
        switch (sort_mode_) {
            case LatencySortMode::ByLatency:
                std::sort(stats_.begin(), stats_.end(),
                    [this](const LayerStats& a, const LayerStats& b) {
                        double va = use_moving_avg_ ? a.movingAverageUs()
                                                    : static_cast<double>(a.last_us);
                        double vb = use_moving_avg_ ? b.movingAverageUs()
                                                    : static_cast<double>(b.last_us);
                        return va > vb;  // descending — slowest first
                    });
                break;

            case LatencySortMode::ByLayerOrder:
                std::sort(stats_.begin(), stats_.end(),
                    [](const LayerStats& a, const LayerStats& b) {
                        return a.layer_name < b.layer_name;
                    });
                break;

            case LatencySortMode::ByName:
                std::sort(stats_.begin(), stats_.end(),
                    [](const LayerStats& a, const LayerStats& b) {
                        return a.layer_name < b.layer_name;
                    });
                break;
        }
    }



    double computeMaxUs() const {
        double max_val = 1.0;
        for (const auto& s : stats_) {
            double v = use_moving_avg_
                ? s.movingAverageUs()
                : static_cast<double>(s.last_us);
            if (v > max_val) max_val = v;
        }
        return max_val;
    }

    static int barColorPair(float fraction) {
        if (fraction > 0.66f) return colorPairBarHigh();
        if (fraction > 0.33f) return colorPairBarMid();
        return colorPairBarLow();
    }

    void clampScroll() {
        if (stats_.empty()) { scroll_top_ = 0; return; }
        int visible = std::max(1, rect_.innerHeight() - 2);
        if (stats_.size() <= (size_t)visible) {
            scroll_top_ = 0;
        } else if (scroll_top_ > stats_.size() - visible) {
            scroll_top_ = stats_.size() - visible;
        }
    }



    const LatencyTracker&    tracker_;
    std::vector<LayerStats>  stats_;         // Snapshot sorted each frame
    LatencySortMode          sort_mode_;
    bool                     use_moving_avg_;
    size_t                   scroll_top_;
};

} // namespace tui
} // namespace llmtracer