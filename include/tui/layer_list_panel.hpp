#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>

#include "panel_base.hpp"
#include "tui/panel_base.hpp"
#include "core/layer_record.hpp"
#include "core/ring_buffer.hpp"
#include "utils/math_utils.hpp"

namespace llmtracer {
namespace tui {


class LayerListPanel : public PanelBase {
public:
    LayerListPanel() : PanelBase(PanelId::LayerList)
                     , cursor_(0)
                     , scroll_top_(0)
                     , auto_follow_(true)
                     , last_buffer_count_(0)
    {}

    // ── PanelBase interface ───────────────────────────────────────────────────

    void init(const Rect& rect) override {
        PanelBase::init(rect);
    }

    void resize(const Rect& rect) override {
        PanelBase::resize(rect);
        clampScroll();
    }

    void render(const RenderState& state) override {
        if (!win_) return;


        state.buffer.snapshot(records_);


        size_t current_count = records_.size();
        if (auto_follow_ && current_count > last_buffer_count_ && !records_.empty()) {
            cursor_ = current_count - 1;
            clampScroll();
        }
        last_buffer_count_ = current_count;

        drawBorder(buildTitle(state));
        drawRecords(state);
        drawScrollBar();
        drawFooter(state);

        refresh();
    }

    bool onKey(const KeyEvent& key) override {
        if (records_.empty()) return false;

        size_t count = records_.size();

        if (key.isDown()) {
            if (cursor_ + 1 < count) {
                ++cursor_;
                auto_follow_ = false;
                clampScroll();
            }
            return true;
        }
        if (key.isUp()) {
            if (cursor_ > 0) {
                --cursor_;
                auto_follow_ = false;
                clampScroll();
            }
            return true;
        }
        if (key.isPageDown()) {
            size_t page = static_cast<size_t>(std::max(1, rect_.innerHeight() - 1));
            cursor_ = std::min(count - 1, cursor_ + page);
            auto_follow_ = false;
            clampScroll();
            return true;
        }
        if (key.isPageUp()) {
            size_t page = static_cast<size_t>(std::max(1, rect_.innerHeight() - 1));
            cursor_ = (cursor_ >= page) ? cursor_ - page : 0;
            auto_follow_ = false;
            clampScroll();
            return true;
        }
        if (key.isHome()) {
            cursor_      = 0;
            scroll_top_  = 0;
            auto_follow_ = false;
            return true;
        }
        if (key.isEnd()) {
            cursor_      = count - 1;
            auto_follow_ = false;
            clampScroll();
            return true;
        }
        if (key.raw == 'f' || key.raw == 'F') {
            // Toggle auto-follow; if turning on, jump to newest immediately
            auto_follow_ = !auto_follow_;
            if (auto_follow_ && !records_.empty()) {
                cursor_ = records_.size() - 1;
                clampScroll();
            }
            return true;
        }

        if (key.isSelect()) {
            return true;
        }

        return false;
    }

    std::string statusHint() const override {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "Layers  j/k:scroll  Space:select  g/G:top/bottom  f:auto-follow[%s]"
            "  Tab:switch panel",
            auto_follow_ ? "ON" : "OFF");
        return std::string(buf);
    }


    std::optional<LayerRecord> selectedRecord() const {
        if (records_.empty() || cursor_ >= records_.size()) return std::nullopt;
        return records_[cursor_];
    }

    size_t selectedIndex() const { return cursor_; }

private:


    std::string buildTitle(const RenderState& state) const {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
                      " Layers [%zu/%zu] ",
                      records_.empty() ? 0 : cursor_ + 1,
                      records_.size());
        if (state.paused) {
            return std::string(buf) + "⏸ ";
        }
        return std::string(buf);
    }

    void drawRecords(const RenderState& /*state*/) {
        if (records_.empty()) {
            writeAt(0, 0, "  No layers captured yet.", COLOR_PAIR(colorPairMuted()));
            writeAt(2, 0, "  Waiting for model forward", COLOR_PAIR(colorPairMuted()));
            writeAt(3, 0, "  pass...", COLOR_PAIR(colorPairMuted()));
            return;
        }

        int visible_rows = rect_.innerHeight() - 1;  // reserve 1 row for footer
        if (visible_rows <= 0) return;

        for (int row = 0; row < visible_rows; ++row) {
            size_t record_idx = scroll_top_ + static_cast<size_t>(row);
            if (record_idx >= records_.size()) {
                clearRow(row);
                continue;
            }

            const LayerRecord& rec   = records_[record_idx];
            bool               is_selected = (record_idx == cursor_);

            drawRow(row, rec, is_selected);
        }
    }

    void drawRow(int row, const LayerRecord& rec, bool selected) {
        int inner_w = rect_.innerWidth();
        if (inner_w < 12) return;


        const char* cursor_char = selected ? "▶" : " ";


        const char* badge_str  = layerTypeToString(rec.layer_type);
        int         badge_pair = layerBadgeColorPair(rec.layer_type);


        std::string lat_str = math::formatLatency(rec.latencyMicros());

        while ((int)lat_str.size() < 8) lat_str = " " + lat_str;
        if ((int)lat_str.size() > 8) lat_str = lat_str.substr(0, 8);


        bool anomaly = rec.output_stats.isAnomolous() || rec.input_stats.isAnomolous();
        const char* flag_str = anomaly ? "⚠ " : "  ";

        int fixed_chars = 1 + 1 + 5 + 1 + 8 + 2;
        int name_width  = std::max(4, inner_w - fixed_chars);

        std::string name = rec.layer_name;
        // Truncate from the left (show the end of the name — most distinctive part)
        if ((int)name.size() > name_width) {
            name = "…" + name.substr(name.size() - (name_width - 1));
        }
        // Pad to name_width
        while ((int)name.size() < name_width) name += ' ';


        if (selected) {
            // Highlight entire row
            wattron(win_, COLOR_PAIR(colorPairSelected()) | A_BOLD);
            // Fill the whole inner row with spaces first
            int abs_row = rect_.innerRow() + row;
            wmove(win_, abs_row, rect_.innerCol());
            for (int c = 0; c < inner_w; ++c) waddch(win_, ' ');
            wattroff(win_, COLOR_PAIR(colorPairSelected()) | A_BOLD);
        }

        int col = 0;

        // Cursor char
        {
            int pair = selected ? colorPairSelected() : colorPairMuted();
            int attr = selected ? (COLOR_PAIR(pair) | A_BOLD) : COLOR_PAIR(pair);
            writeAt(row, col, cursor_char, attr);
            col += 1;
        }

        // Space
        writeAt(row, col, " ");
        col += 1;

        // Badge
        {
            int attr = COLOR_PAIR(badge_pair) | A_BOLD;
            writeAt(row, col, badge_str, attr);
            col += 5;
        }

        // Space
        writeAt(row, col, " ");
        col += 1;

        // Layer name
        {
            int pair = selected ? colorPairSelected() : 0;
            int attr = selected ? (COLOR_PAIR(pair) | A_BOLD) : 0;
            writeAt(row, col, name, attr);
            col += name_width;
        }

        // Latency
        {
            int lat_pair = selected ? colorPairSelected() : colorPairMuted();
            writeAt(row, col, lat_str, COLOR_PAIR(lat_pair));
            col += 8;
        }

        // Anomaly flag
        {
            int pair = anomaly ? colorPairWarn() : colorPairMuted();
            writeAt(row, col, flag_str, COLOR_PAIR(pair));
        }
    }

    // Thin scrollbar on the right edge of the inner area
    void drawScrollBar() {
        if (records_.empty()) return;
        int inner_h = rect_.innerHeight() - 1;  // match visible_rows
        if (inner_h <= 0) return;

        float fill_start = static_cast<float>(scroll_top_)
                         / static_cast<float>(records_.size());
        float fill_end   = static_cast<float>(scroll_top_ + inner_h)
                         / static_cast<float>(records_.size());
        fill_end = std::min(fill_end, 1.0f);

        int bar_start = static_cast<int>(fill_start * inner_h);
        int bar_end   = static_cast<int>(fill_end   * inner_h);

        int scroll_col = rect_.innerWidth() - 1;  // Rightmost inner column

        for (int r = 0; r < inner_h; ++r) {
            bool in_bar = (r >= bar_start && r <= bar_end);
            const char* ch = in_bar ? "▓" : "░";
            int abs_row = rect_.innerRow() + r;
            int abs_col = rect_.innerCol() + scroll_col;
            wattron(win_, COLOR_PAIR(colorPairMuted()));
            mvwaddstr(win_, abs_row, abs_col, ch);
            wattroff(win_, COLOR_PAIR(colorPairMuted()));
        }
    }

    // One-line footer showing buffer fill and auto-follow state
    void drawFooter(const RenderState& state) {
        int footer_row = rect_.innerHeight() - 1;
        if (footer_row < 0) return;

        char buf[128];
        float fill = state.buffer.fillRatio() * 100.0f;
        std::snprintf(buf, sizeof(buf),
                      " buf:%.0f%%  %s",
                      fill,
                      auto_follow_ ? "⟳follow" : "⏸frozen");

        int attr = COLOR_PAIR(colorPairMuted());
        writeAt(footer_row, 0, std::string(buf), attr);
    }


    void clampScroll() {
        if (records_.empty()) { scroll_top_ = 0; cursor_ = 0; return; }

        // Clamp cursor to valid range
        if (cursor_ >= records_.size()) cursor_ = records_.size() - 1;

        int visible_rows = std::max(1, rect_.innerHeight() - 1);

        // Scroll down if cursor is below the visible window
        if (cursor_ >= scroll_top_ + static_cast<size_t>(visible_rows)) {
            scroll_top_ = cursor_ - static_cast<size_t>(visible_rows) + 1;
        }
        // Scroll up if cursor is above the visible window
        if (cursor_ < scroll_top_) {
            scroll_top_ = cursor_;
        }
    }



    static int layerBadgeColorPair(LayerType t) {
        switch (t) {
            case LayerType::Embedding:  return colorPairEmbed();
            case LayerType::Attention:  return colorPairAttn();
            case LayerType::MLP:        return colorPairMlp();
            case LayerType::LayerNorm:  return colorPairNorm();
            case LayerType::Output:     return colorPairOut();
            default:                    return colorPairMuted();
        }
    }



    std::vector<LayerRecord> records_;       // Snapshot of ring buffer contents
    size_t                   cursor_;        // Index of highlighted row
    size_t                   scroll_top_;    // Index of first visible row
    bool                     auto_follow_;   // Automatically track newest record
    size_t                   last_buffer_count_; // For detecting new arrivals
};

} // namespace tui
} // namespace llmtracer