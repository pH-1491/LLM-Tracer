#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "panel_base.hpp"
#include "tui/panel_base.hpp"
#include "core/layer_record.hpp"
#include "utils/color.hpp"
#include "utils/math_utils.hpp"

namespace llmtracer {
namespace tui {


class AttentionPanel : public PanelBase {
public:
    AttentionPanel()
        : PanelBase(PanelId::Attention)
        , current_head_(0)
        , show_mean_heads_(false)
        , last_record_id_(0)
    {}



    void init(const Rect& rect) override {
        PanelBase::init(rect);
    }

    void resize(const Rect& rect) override {
        PanelBase::resize(rect);
    }

    void render(const RenderState& state) override {
        if (!win_) return;


        if (!state.selected_record.has_value()) {
            drawBorder(" Attention ");
            writeAt(1, 1, "Select an Attention layer",
                    COLOR_PAIR(colorPairMuted()));
            writeAt(2, 1, "to view its heatmap.",
                    COLOR_PAIR(colorPairMuted()));
            refresh();
            return;
        }

        const LayerRecord& rec = state.selected_record.value();

        if (rec.layer_type != LayerType::Attention || !rec.attention.isValid()) {
            drawBorder(" Attention ");
            writeAt(1, 1, "Selected layer is not an",
                    COLOR_PAIR(colorPairMuted()));
            writeAt(2, 1, "Attention layer.",
                    COLOR_PAIR(colorPairMuted()));
            writeAt(3, 1, std::string("Type: ") + layerTypeToString(rec.layer_type),
                    COLOR_PAIR(colorPairMuted()));
            refresh();
            return;
        }


        if (rec.record_id != last_record_id_) {
            last_record_id_ = rec.record_id;
            // Clamp head to valid range for this record
            current_head_ = std::min(current_head_,
                                     std::max(0, rec.attention.num_heads - 1));
        }


        std::string title = buildTitle(rec);
        drawBorder(title);
        drawHeatmap(rec);
        drawLegend();
        drawFooterHints();

        refresh();
    }

    bool onKey(const KeyEvent& key) override {
        // Head cycling
        if (key.isHeadNext() || key.isRight()) {
            ++current_head_;
            return true;
        }
        if (key.isHeadPrev() || key.isLeft()) {
            if (current_head_ > 0) --current_head_;
            return true;
        }
        // Toggle mean-heads view
        if (key.isToggle()) {
            show_mean_heads_ = !show_mean_heads_;
            return true;
        }
        return false;
    }

    std::string statusHint() const override {
        return "Attention  [/]: prev/next head  t: toggle avg-heads  Tab: switch panel";
    }

private:

    std::string buildTitle(const LayerRecord& rec) const {
        char buf[64];
        if (show_mean_heads_) {
            std::snprintf(buf, sizeof(buf),
                          " Attention  avg/%d heads  [%d×%d] ",
                          rec.attention.num_heads,
                          rec.attention.seq_len,
                          rec.attention.seq_len);
        } else {
            std::snprintf(buf, sizeof(buf),
                          " Attention  head %d/%d  [%d×%d] ",
                          current_head_,
                          rec.attention.num_heads,
                          rec.attention.seq_len,
                          rec.attention.seq_len);
        }
        return std::string(buf);
    }



    void drawHeatmap(const LayerRecord& rec) {
        const AttentionData& attn = rec.attention;

        // Clamp head index now that we have num_heads
        if (current_head_ >= attn.num_heads) {
            current_head_ = std::max(0, attn.num_heads - 1);
        }

        int inner_h = rect_.innerHeight() - 2;  // reserve 2 rows: col-header + footer
        int inner_w = rect_.innerWidth()  - 5;  // reserve 5 cols for row labels

        if (inner_h <= 0 || inner_w <= 0) return;


        int display_seq_col = std::min(attn.seq_len, inner_w / 2);
        int display_seq_row = std::min(attn.seq_len, inner_h - 1);


        const std::vector<float>& matrix = attn.matrix;


        {
            std::string header = "Q\\K ";
            for (int c = 0; c < display_seq_col; ++c) {
                char cell[4];
                // Each column label is 2 chars wide to match cell width
                std::snprintf(cell, sizeof(cell), "%2d", c % 100);
                header += cell;
            }
            writeAt(0, 1, header, COLOR_PAIR(colorPairMuted()) | A_BOLD);
        }


        for (int r = 0; r < display_seq_row; ++r) {
            int content_row = r + 1;  // offset by header row
            if (content_row >= inner_h) break;

            // Row label (right-aligned 3 chars + space)
            char row_label[8];
            std::snprintf(row_label, sizeof(row_label), "%3d ", r);
            writeAt(content_row, 1, std::string(row_label),
                    COLOR_PAIR(colorPairMuted()));

            // Each cell in this row
            for (int c = 0; c < display_seq_col; ++c) {
                float weight = attn.at(r, c);  // already normalised [0,1]

                // Map weight to one of 5 heat colour stops
                uint8_t bg_color = color::heatColor(weight);

                // Choose foreground char based on weight intensity
                // High weight: solid block; low weight: light shade
                const char* cell_char = heatChar(weight);

                std::string cell_str =
                    color::bg(bg_color) +
                    color::fg(heatFg(weight)) +
                    cell_char +
                    cell_char +      // 2-wide cell
                    color::reset();

                // Calculate absolute position in the window
                int abs_row = rect_.innerRow() + content_row;
                int abs_col = rect_.innerCol() + 5 + c * 2;  // 5 = row label width

                if (abs_row < rect_.row + rect_.height - 1 &&
                    abs_col < rect_.col + rect_.width  - 1)
                {
                    mvwaddstr(win_, abs_row, abs_col, cell_str.c_str());
                }
            }
        }
    }


    static const char* heatChar(float w) {
        if (w < 0.15f) return " ";   // Near-zero: empty (background colour shows)
        if (w < 0.35f) return "░";   // Low
        if (w < 0.55f) return "▒";   // Medium-low
        if (w < 0.75f) return "▓";   // Medium-high
        return "█";                   // High: solid block
    }


    static uint8_t heatFg(float w) {
        if (w < 0.4f) return 240;   // dark bg → light grey fg
        if (w < 0.7f) return 255;   // mid bg  → white fg
        return 16;                   // hot bg  → black fg (contrast on red)
    }



    void drawLegend() {
        int legend_row = rect_.innerHeight() - 2;
        if (legend_row < 1) return;

        int inner_w = rect_.innerWidth() - 2;
        if (inner_w < 20) return;

        // Build legend string: "low " + gradient cells + " high"
        std::string legend = "low ";
        int grad_cells = std::min(inner_w - 12, 20);
        for (int i = 0; i < grad_cells; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(grad_cells - 1);
            uint8_t bg = color::heatColor(t);
            legend += color::bg(bg) + " " + color::reset();
        }
        legend += " high";

        // Pad to inner width
        int raw_len = 4 + grad_cells + 5;  // "low " + cells + " high"
        int pad = inner_w - raw_len;
        if (pad > 0) legend += std::string(pad, ' ');

        int abs_row = rect_.innerRow() + legend_row;
        int abs_col = rect_.innerCol() + 1;
        wattron(win_, COLOR_PAIR(colorPairMuted()));
        mvwaddstr(win_, abs_row, abs_col, legend.c_str());
        wattroff(win_, COLOR_PAIR(colorPairMuted()));
    }


    void drawFooterHints() {
        int hint_row = rect_.innerHeight() - 1;
        if (hint_row < 1) return;

        std::string hint = show_mean_heads_
            ? " [t]:single-head  []/h/l:cycle head"
            : " [t]:avg-heads    []/h/l:cycle head";

        int w = rect_.innerWidth() - 2;
        if ((int)hint.size() > w) hint = hint.substr(0, w);

        writeAt(hint_row, 1, hint, COLOR_PAIR(colorPairMuted()));
    }



    int      current_head_;       // Which attention head is displayed
    bool     show_mean_heads_;    // Average all heads instead of single
    uint64_t last_record_id_;     // Detect record changes for head clamping
};

} // namespace tui
} // namespace llmtracer