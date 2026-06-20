#pragma once

#include <string>
#include <cstdint>
#include <cstdio>
#include <cmath>


namespace llmtracer {
namespace color {



inline const char* reset() { return "\033[0m"; }

inline const char* bold()  { return "\033[1m"; }
inline const char* dim()   { return "\033[2m"; }


inline const char* underline() { return "\033[4m"; }

inline const char* reverse() { return "\033[7m"; }

namespace Palette {

    // ── Structural UI colors ──────────────────────────────────────────────────
    constexpr uint8_t kBorder       = 240;  // Dark grey  — panel borders
    constexpr uint8_t kTitle        = 255;  // Bright white — panel titles
    constexpr uint8_t kSubtle       = 238;  // Mid grey   — secondary labels
    constexpr uint8_t kSelected     = 39;   // Bright cyan — selected row
    constexpr uint8_t kFocused      = 33;   // Blue        — focused panel border
    constexpr uint8_t kMuted        = 243;  // Grey        — unfocused text

    // ── Layer type badge colors ───────────────────────────────────────────────
    constexpr uint8_t kEmbedding    = 141;  // Soft purple  — Embedding layers
    constexpr uint8_t kAttention    = 75;   // Sky blue     — Attention layers
    constexpr uint8_t kMLP          = 114;  // Mint green   — MLP/FFN layers
    constexpr uint8_t kLayerNorm    = 179;  // Amber        — LayerNorm
    constexpr uint8_t kOutput       = 210;  // Salmon pink  — Output projection
    constexpr uint8_t kUnknown      = 243;  // Grey         — Unknown layers

    // ── Status / anomaly colors ───────────────────────────────────────────────
    constexpr uint8_t kOk           = 114;  // Green  — healthy values
    constexpr uint8_t kWarn         = 221;  // Yellow — elevated sparsity / large values
    constexpr uint8_t kError        = 196;  // Red    — NaN / Inf detected
    constexpr uint8_t kInfo         = 39;   // Cyan   — informational highlights

    // ── Latency bar chart ─────────────────────────────────────────────────────
    constexpr uint8_t kBarLow       = 34;   // Green  — fast layers (< 33% of max)
    constexpr uint8_t kBarMid       = 220;  // Yellow — mid layers  (33–66%)
    constexpr uint8_t kBarHigh      = 196;  // Red    — slow layers (> 66% of max)
    constexpr uint8_t kBarBg        = 236;  // Very dark grey — bar background fill

    // ── Attention heatmap gradient (cold → hot, 5 stops) ─────────────────────
    // Maps normalized attention weight [0.0, 1.0] → one of 5 color stops.
    // Stop 0 = near zero attention (cold), Stop 4 = maximum attention (hot).
    constexpr uint8_t kHeat[5]      = { 17, 20, 27, 208, 196 };
    //                                  dark navy → blue → cyan → orange → red

}


// Set foreground color to a 256-color palette index
inline std::string fg(uint8_t idx) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "\033[38;5;%dm", idx);
    return std::string(buf);
}

// Set background color to a 256-color palette index
inline std::string bg(uint8_t idx) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "\033[48;5;%dm", idx);
    return std::string(buf);
}

// Convenience: set both fg and bg in one call
inline std::string fgbg(uint8_t fg_idx, uint8_t bg_idx) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "\033[38;5;%d;48;5;%dm", fg_idx, bg_idx);
    return std::string(buf);
}



inline uint8_t layerTypeBadgeColor(LayerType t) {

    switch (t) {
        case LayerType::Embedding:  return Palette::kEmbedding;
        case LayerType::Attention:  return Palette::kAttention;
        case LayerType::MLP:        return Palette::kMLP;
        case LayerType::LayerNorm:  return Palette::kLayerNorm;
        case LayerType::Output:     return Palette::kOutput;
        default:                    return Palette::kUnknown;
    }
}

// Color for a stat value based on whether it's anomalous
inline uint8_t statValueColor(const ActivationStats& s) {
    if (s.has_nan || s.has_inf)  return Palette::kError;
    if (s.sparsity > 0.9f)       return Palette::kWarn;
    return Palette::kOk;
}

// Color for a latency bar given this layer's fraction of the worst layer's time.
// fraction is in [0.0, 1.0] where 1.0 = the slowest layer.
inline uint8_t latencyBarColor(float fraction) {
    if (fraction > 0.66f) return Palette::kBarHigh;
    if (fraction > 0.33f) return Palette::kBarMid;
    return Palette::kBarLow;
}



inline uint8_t heatColor(float normalized) {
    // Clamp to valid range
    if (normalized <= 0.0f) return Palette::kHeat[0];
    if (normalized >= 1.0f) return Palette::kHeat[4];

    // Map to one of 5 buckets
    int idx = static_cast<int>(normalized * 4.999f);  // 0..4
    return Palette::kHeat[idx];
}



// Colored foreground text, auto-reset after
inline std::string colored(const std::string& text, uint8_t fg_idx) {
    return fg(fg_idx) + text + reset();
}

// Bold colored text
inline std::string boldColored(const std::string& text, uint8_t fg_idx) {
    return std::string(bold()) + fg(fg_idx) + text + reset();
}

// Highlighted text: colored background with white foreground
inline std::string highlighted(const std::string& text,
                                uint8_t            bg_idx,
                                uint8_t            fg_idx = 255)
{
    return fgbg(fg_idx, bg_idx) + text + reset();
}


struct BoxChars {
    const char* tl;   // Top-left corner
    const char* tr;   // Top-right corner
    const char* bl;   // Bottom-left corner
    const char* br;   // Bottom-right corner
    const char* h;    // Horizontal line
    const char* v;    // Vertical line
    const char* lt;   // Left T-junction  (├)
    const char* rt;   // Right T-junction (┤)
    const char* tt;   // Top T-junction   (┬)
    const char* bt;   // Bottom T-junction(┴)
    const char* cross;// Cross            (┼)
};

// Rounded corners — used for main panel borders
inline BoxChars roundedBox() {
    return { "╭", "╮", "╰", "╯", "─", "│", "├", "┤", "┬", "┴", "┼" };
}

// Sharp corners — used for inner dividers and sub-sections
inline BoxChars sharpBox() {
    return { "┌", "┐", "└", "┘", "─", "│", "├", "┤", "┬", "┴", "┼" };
}

// Heavy borders — used for the focused (active) panel
inline BoxChars heavyBox() {
    return { "┏", "┓", "┗", "┛", "━", "┃", "┣", "┫", "┳", "┻", "╋" };
}



inline const char* fractionalBlock(float fraction) {
    static const char* blocks[] = {
        " ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"
    };
    int idx = static_cast<int>(fraction * 8.0f);
    if (idx < 0) idx = 0;
    if (idx > 8) idx = 8;
    return blocks[idx];
}


inline const char* fullBlock() { return "█"; }


inline const char* dimBlock()  { return "░"; }

inline const char* clearLine() { return "\033[K"; }

// Move cursor to (row, col) — 1-indexed, used outside of ncurses context
inline std::string moveCursor(int row, int col) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "\033[%d;%dH", row, col);
    return std::string(buf);
}

}
}