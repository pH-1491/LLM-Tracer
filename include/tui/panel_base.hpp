#pragma once

#include <string>
#include <cstdint>
#include <functional>

// ncurses — the terminal UI library.
// Must be included before any use of WINDOW*, waddstr, etc.
// On Linux: link with -lncurses (or -lncursesw for wide-char support)
#include <ncurses.h>
#include <optional>

#include "core/layer_record.hpp"
#include "core/ring_buffer.hpp"

namespace llmtracer {
    struct LayerRecord;
    class RingBufferView;
}

namespace llmtracer {
namespace tui {

struct Rect {
    int row    = 0;   // Top edge (0 = top of terminal)
    int col    = 0;   // Left edge (0 = left of terminal)
    int height = 0;   // Number of rows
    int width  = 0;   // Number of columns

    // Inner content area (excluding 1-cell border)
    int innerRow()    const { return row + 1; }
    int innerCol()    const { return col + 1; }
    int innerHeight() const { return std::max(0, height - 2); }
    int innerWidth()  const { return std::max(0, width  - 2); }

    bool isValid()    const { return height > 2 && width > 4; }
};


struct KeyEvent {
    int  raw;           // Raw ncurses return value from getch()

    // Convenience predicates — panels use these instead of comparing raw values
    bool isUp()     const { return raw == KEY_UP    || raw == 'k'; }
    bool isDown()   const { return raw == KEY_DOWN  || raw == 'j'; }
    bool isLeft()   const { return raw == KEY_LEFT  || raw == 'h'; }
    bool isRight()  const { return raw == KEY_RIGHT || raw == 'l'; }
    bool isSelect() const { return raw == ' ' || raw == '\n' || raw == '\r'; }
    bool isTab()    const { return raw == '\t'; }
    bool isEscape() const { return raw == 27; }
    bool isQuit()   const { return raw == 'q' || raw == 'Q'; }
    bool isPause()  const { return raw == 'p' || raw == 'P'; }
    bool isReset()  const { return raw == 'r' || raw == 'R'; }
    bool isHelp()   const { return raw == '?' || raw == 'H'; }

    // Page up/down for long lists
    bool isPageUp()   const { return raw == KEY_PPAGE || raw == 'u'; }
    bool isPageDown() const { return raw == KEY_NPAGE || raw == 'd'; }

    // Home / End — jump to first/last item
    bool isHome() const { return raw == KEY_HOME || raw == 'g'; }
    bool isEnd()  const { return raw == KEY_END  || raw == 'G'; }

    // Cycle heads in attention panel
    bool isHeadNext() const { return raw == ']'; }
    bool isHeadPrev() const { return raw == '['; }

    // Toggle between views within a panel
    bool isToggle() const { return raw == 't' || raw == 'T'; }
};


enum class PanelId {
    LayerList   = 0,    // Left panel — scrollable list of layers
    Metrics     = 1,    // Top-right — tensor shapes & activation stats
    Attention   = 2,    // Mid-right — attention heatmap
    Latency     = 3,    // Bottom-right — latency bar chart
    Count       = 4     // Sentinel — number of panels (used for Tab cycling)
};

inline const char* panelIdName(PanelId id) {
    switch (id) {
        case PanelId::LayerList: return "Layers";
        case PanelId::Metrics:   return "Metrics";
        case PanelId::Attention: return "Attention";
        case PanelId::Latency:   return "Latency";
        default:                  return "Unknown";
    }
}


struct RenderState {
    const RingBufferView& buffer;       // All captured layer records
    bool                  paused;       // True when user has frozen the display
    uint64_t              frame_count;  // Monotonically increasing render counter
    int                   terminal_rows;
    int                   terminal_cols;

    // The record currently selected in the LayerList panel.
    // Other panels use this to determine what to display.
    std::optional<LayerRecord> selected_record;

    // Index of selected item in the LayerList (for scroll position sync)
    size_t selected_index = 0;
};


class PanelBase {
public:
    explicit PanelBase(PanelId id) : id_(id), focused_(false), win_(nullptr) {}

    virtual ~PanelBase() {
        if (win_) {
            delwin(win_);
            win_ = nullptr;
        }
    }

    // Non-copyable
    PanelBase(const PanelBase&)            = delete;
    PanelBase& operator=(const PanelBase&) = delete;


    // Called once after ncurses is initialised.
    // Subclasses should allocate their WINDOW* here.
    virtual void init(const Rect& initial_rect) {
        rect_ = initial_rect;
        recreateWindow();
    }

    // Called when the terminal is resized.
    // Subclasses must handle the new dimensions and recreate their WINDOW*.
    virtual void resize(const Rect& new_rect) {
        rect_ = new_rect;
        recreateWindow();
    }

    // Called on every render frame.
    // Subclasses draw their content into win_ then call wrefresh(win_).
    virtual void render(const RenderState& state) = 0;

    // Called when a keyboard event reaches this panel (only when focused).
    // Returns true if the event was consumed, false to propagate it up.
    virtual bool onKey(const KeyEvent& key) = 0;

    // Called before ncurses teardown.
    virtual void shutdown() {
        if (win_) {
            delwin(win_);
            win_ = nullptr;
        }
    }



    void setFocused(bool focused) {
        bool changed = (focused != focused_);
        focused_ = focused;
        if (changed && win_) {
            // Force a redraw so border color updates immediately
            touchwin(win_);
        }
    }

    bool isFocused() const { return focused_; }



    PanelId     id()   const { return id_; }
    const Rect& rect() const { return rect_; }
    const char* name() const { return panelIdName(id_); }


    // Returned string is shown in the TuiApp's global status bar when this
    // panel is focused.
    virtual std::string statusHint() const {
        return std::string(name()) + " panel  |  Tab: switch focus  |  q: quit";
    }

protected:

    void recreateWindow() {
        if (win_) {
            delwin(win_);
            win_ = nullptr;
        }
        if (rect_.isValid()) {
            win_ = newwin(rect_.height, rect_.width, rect_.row, rect_.col);
            if (win_) {
                keypad(win_, TRUE);     // Enable function-key input in this window
                nodelay(win_, TRUE);    // Non-blocking getch on this window
            }
        }
    }

    // Draw a bordered box around the panel with a title.
    // Uses heavy borders when focused, rounded when not.
    // Must be called at the start of render() before drawing content.
    void drawBorder(const std::string& title = "") const {
        if (!win_) return;

        werase(win_);


        if (focused_) {
            // Heavy border
            wattron(win_, COLOR_PAIR(colorPairFocused()));
            box(win_, 0, 0);   // Uses default box chars; we'll override below
            // Draw heavy corners manually
            mvwaddstr(win_, 0,              0,               "┏");
            mvwaddstr(win_, 0,              rect_.width - 1, "┓");
            mvwaddstr(win_, rect_.height-1, 0,               "┗");
            mvwaddstr(win_, rect_.height-1, rect_.width - 1, "┛");
            // Horizontal edges
            for (int c = 1; c < rect_.width - 1; ++c) {
                mvwaddstr(win_, 0,              c, "━");
                mvwaddstr(win_, rect_.height-1, c, "━");
            }
            // Vertical edges
            for (int r = 1; r < rect_.height - 1; ++r) {
                mvwaddstr(win_, r, 0,               "┃");
                mvwaddstr(win_, r, rect_.width - 1, "┃");
            }
            wattroff(win_, COLOR_PAIR(colorPairFocused()));
        } else {
            // Rounded border
            wattron(win_, COLOR_PAIR(colorPairBorder()));
            mvwaddstr(win_, 0,              0,               "╭");
            mvwaddstr(win_, 0,              rect_.width - 1, "╮");
            mvwaddstr(win_, rect_.height-1, 0,               "╰");
            mvwaddstr(win_, rect_.height-1, rect_.width - 1, "╯");
            for (int c = 1; c < rect_.width - 1; ++c) {
                mvwaddstr(win_, 0,              c, "─");
                mvwaddstr(win_, rect_.height-1, c, "─");
            }
            for (int r = 1; r < rect_.height - 1; ++r) {
                mvwaddstr(win_, r, 0,               "│");
                mvwaddstr(win_, r, rect_.width - 1, "│");
            }
            wattroff(win_, COLOR_PAIR(colorPairBorder()));
        }

        // Draw title centred in the top border
        if (!title.empty()) {
            int max_title = rect_.width - 4;
            std::string t = " " + title + " ";
            if ((int)t.size() > max_title) t = t.substr(0, max_title);
            int title_col = (rect_.width - static_cast<int>(t.size())) / 2;
            wattron(win_, COLOR_PAIR(colorPairTitle()) | A_BOLD);
            mvwaddstr(win_, 0, title_col, t.c_str());
            wattroff(win_, COLOR_PAIR(colorPairTitle()) | A_BOLD);
        }
    }

    // Write a string clipped to the inner width.
    // row, col are relative to the INNER content area (not the window origin).
    void writeAt(int row, int col, const std::string& text,
                 int attr = 0) const
    {
        if (!win_) return;
        int abs_row = rect_.innerRow()    + row;
        int abs_col = rect_.innerCol()    + col;
        int max_col = rect_.col + rect_.width - 2;   // Right border column
        if (abs_row >= rect_.row + rect_.height - 1) return;  // Below bottom border
        if (abs_col >= max_col) return;

        int avail = max_col - abs_col;
        if (avail <= 0) return;

        std::string clipped = text.substr(0, static_cast<size_t>(avail));

        if (attr) wattron(win_, attr);
        mvwaddstr(win_, abs_row, abs_col, clipped.c_str());
        if (attr) wattroff(win_, attr);
    }

    // Fill a row of the inner area with spaces (clear that line)
    void clearRow(int inner_row) const {
        if (!win_) return;
        int abs_row = rect_.innerRow() + inner_row;
        if (abs_row >= rect_.row + rect_.height - 1) return;
        wmove(win_, abs_row, rect_.innerCol());
        for (int c = 0; c < rect_.innerWidth(); ++c) waddch(win_, ' ');
    }

    // Refresh this panel's window to the terminal
    void refresh() const {
        if (win_) wrefresh(win_);
    }

    
    static int colorPairBorder()   { return 1; }
    static int colorPairFocused()  { return 2; }
    static int colorPairTitle()    { return 3; }
    static int colorPairSelected() { return 4; }
    static int colorPairOk()       { return 5; }
    static int colorPairWarn()     { return 6; }
    static int colorPairError()    { return 7; }
    static int colorPairMuted()    { return 8; }
    static int colorPairAttn()     { return 9; }  // Attention layer badge
    static int colorPairMlp()      { return 10; } // MLP layer badge
    static int colorPairEmbed()    { return 11; } // Embedding badge
    static int colorPairNorm()     { return 12; } // LayerNorm badge
    static int colorPairOut()      { return 13; } // Output badge
    static int colorPairBarLow()   { return 14; } // Fast latency bar
    static int colorPairBarMid()   { return 15; } // Medium latency bar
    static int colorPairBarHigh()  { return 16; } // Slow latency bar

    // Member state accessible to subclasses
    PanelId  id_;
    bool     focused_;
    Rect     rect_;
    WINDOW*  win_;
};

} // namespace tui
} // namespace llmtracer