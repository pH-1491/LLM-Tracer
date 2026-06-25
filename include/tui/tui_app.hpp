#pragma once

#include <memory>
#include <array>
#include <string>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <thread>

// ncurses — must be included before any use of initscr(), COLOR_PAIR(), etc.
#include <ncurses.h>

#include "panel_base.hpp"
#include "core/ring_buffer.hpp"
#include "instrumentation/latency_tracker.hpp"
#include "tui/panel_base.hpp"
#include "tui/layer_list_panel.hpp"
#include "tui/metrics_panel.hpp"
#include "tui/attention_panel.hpp"
#include "tui/latency_panel.hpp"
#include "utils/color.hpp"

namespace llmtracer {
namespace tui {


struct TuiConfig {
    int    target_fps         = 10;

    // Left panel width as a fraction of total terminal width [0.2, 0.6]
    float  left_panel_ratio   = 0.35f;

    // Right panel height splits (fractions of right-column height)
    float  metrics_ratio      = 0.35f;   // metrics panel
    float  attention_ratio    = 0.40f;   // attention panel
    // latency gets the remainder: 1.0 - metrics_ratio - attention_ratio

    // Show a help overlay on first launch
    bool   show_help_on_start = true;
};

class TuiApp {
public:
    TuiApp(RingBuffer&       buffer,
           LatencyTracker&   latency,
           TuiConfig         cfg = {})
        : buffer_view_(buffer)
        , latency_(latency)
        , cfg_(std::move(cfg))
        , focused_panel_(PanelId::LayerList)
        , paused_(false)
        , running_(false)
        , resize_pending_(false)
        , show_help_(cfg_.show_help_on_start)
        , frame_count_(0)
    {
        // Register SIGWINCH handler for terminal resize
        // Uses a global atomic flag that run() checks each frame
        s_resize_flag_.store(false);
        std::signal(SIGWINCH, TuiApp::onSigwinch);
    }

    ~TuiApp() {
        if (running_) shutdown();
    }

    // Non-copyable
    TuiApp(const TuiApp&)            = delete;
    TuiApp& operator=(const TuiApp&) = delete;

    void run() {
        initNcurses();
        initColors();
        constructPanels();
        layoutPanels();
        initPanels();

        running_ = true;

        // Initial render before waiting for input
        renderFrame();

        using Clock    = std::chrono::steady_clock;
        using Ms       = std::chrono::milliseconds;
        const Ms frame_ms{ 1000 / std::max(1, cfg_.target_fps) };

        while (running_) {
            auto frame_start = Clock::now();


            int ch = getch();   // Non-blocking (nodelay was set in initNcurses)
            if (ch != ERR) {
                handleKey(KeyEvent{ ch });
            }


            if (s_resize_flag_.exchange(false)) {
                handleResize();
            }


            if (running_) {
                renderFrame();
            }


            auto elapsed = Clock::now() - frame_start;
            auto sleep   = frame_ms - std::chrono::duration_cast<Ms>(elapsed);
            if (sleep > Ms::zero()) {
                std::this_thread::sleep_for(sleep);
            }
        }

        shutdown();
    }

    // Request a clean exit from outside the render loop
    void requestQuit() {
        running_.store(false);
    }

    bool isPaused() const { return paused_; }

private:
    void initNcurses() {
        initscr();              // Initialise ncurses screen
        cbreak();               // Disable line buffering
        noecho();               // Don't echo typed characters
        keypad(stdscr, TRUE);   // Enable function keys (arrow keys, F1, etc.)
        nodelay(stdscr, TRUE);  // Non-blocking getch() on stdscr
        curs_set(0);            // Hide the text cursor
        set_escdelay(25);       // Short delay for ESC sequences (ms)
    }

    void initColors() {
        if (!has_colors()) return;
        start_color();
        use_default_colors();   // Allow -1 as "terminal default" background



        init_pair(PanelBase::colorPairBorder(),   COLOR_WHITE,   -1);
        init_pair(PanelBase::colorPairFocused(),  COLOR_CYAN,    -1);
        init_pair(PanelBase::colorPairTitle(),    COLOR_WHITE,   -1);
        init_pair(PanelBase::colorPairSelected(), COLOR_BLACK,   COLOR_CYAN);
        init_pair(PanelBase::colorPairOk(),       COLOR_GREEN,   -1);
        init_pair(PanelBase::colorPairWarn(),     COLOR_YELLOW,  -1);
        init_pair(PanelBase::colorPairError(),    COLOR_RED,     -1);
        init_pair(PanelBase::colorPairMuted(),    COLOR_WHITE,   -1);  // dim applied separately

        // Layer type badge colours (256-colour where supported)
        init_pair(PanelBase::colorPairAttn(),     141,           -1);  // purple
        init_pair(PanelBase::colorPairMlp(),      114,           -1);  // mint
        init_pair(PanelBase::colorPairEmbed(),    75,            -1);  // sky blue
        init_pair(PanelBase::colorPairNorm(),     179,           -1);  // amber
        init_pair(PanelBase::colorPairOut(),      210,           -1);  // salmon

        // Latency bar tiers
        init_pair(PanelBase::colorPairBarLow(),   COLOR_GREEN,   -1);
        init_pair(PanelBase::colorPairBarMid(),   COLOR_YELLOW,  -1);
        init_pair(PanelBase::colorPairBarHigh(),  COLOR_RED,     -1);
    }



    void constructPanels() {
        // Panels are stored as polymorphic pointers in a fixed array
        // indexed by PanelId enum value.
        panels_[static_cast<int>(PanelId::LayerList)]
            = std::make_unique<LayerListPanel>();

        panels_[static_cast<int>(PanelId::Metrics)]
            = std::make_unique<MetricsPanel>();

        panels_[static_cast<int>(PanelId::Attention)]
            = std::make_unique<AttentionPanel>();

        panels_[static_cast<int>(PanelId::Latency)]
            = std::make_unique<LatencyPanel>(latency_);
    }

    void initPanels() {
        for (auto& panel : panels_) {
            if (panel) panel->init(panel->rect());
        }
        // Set initial focus
        panels_[static_cast<int>(focused_panel_)]->setFocused(true);
    }



    void layoutPanels() {
        int total_rows, total_cols;
        getmaxyx(stdscr, total_rows, total_cols);

        // Reserve 1 row for the global status bar at the bottom
        int usable_rows = std::max(4, total_rows - 1);

        // Left / right split
        int left_w  = static_cast<int>(total_cols * cfg_.left_panel_ratio);
        left_w      = std::max(20, std::min(left_w, total_cols - 20));
        int right_w = total_cols - left_w;

        // Right column vertical split
        int metrics_h = static_cast<int>(usable_rows * cfg_.metrics_ratio);
        metrics_h     = std::max(6, metrics_h);

        int attn_h    = static_cast<int>(usable_rows * cfg_.attention_ratio);
        attn_h        = std::max(6, attn_h);

        int lat_h     = usable_rows - metrics_h - attn_h;
        lat_h         = std::max(4, lat_h);

        // Assign rects
        setRect(PanelId::LayerList,
                Rect{ 0,      0,      usable_rows, left_w  });
        setRect(PanelId::Metrics,
                Rect{ 0,      left_w, metrics_h,   right_w });
        setRect(PanelId::Attention,
                Rect{ metrics_h,      left_w, attn_h,  right_w });
        setRect(PanelId::Latency,
                Rect{ metrics_h + attn_h, left_w, lat_h,   right_w });

        // Status bar position (stored for drawStatusBar)
        status_bar_row_ = total_rows - 1;
        status_bar_cols_ = total_cols;
    }

    void setRect(PanelId id, const Rect& rect) {
        auto* panel = getPanel(id);
        if (panel) {
            // Store rect on the panel so init() can pick it up
            panel->resize(rect);
        }
    }



    void renderFrame() {
        // Build shared render state
        RenderState state = buildRenderState();

        // Render all panels
        for (auto& panel : panels_) {
            if (panel) panel->render(state);
        }

        // Status bar (rendered directly on stdscr, not a panel)
        drawStatusBar(state);

        // Help overlay (drawn on top of everything else)
        if (show_help_) drawHelpOverlay();

        doupdate();   // Flush all pending window refreshes to terminal at once
        ++frame_count_;
    }

    RenderState buildRenderState() {
        int rows, cols;
        getmaxyx(stdscr, rows, cols);

        RenderState state{
            .buffer         = buffer_view_,
            .paused         = paused_,
            .frame_count    = frame_count_,
            .terminal_rows  = rows,
            .terminal_cols  = cols,
        };

        // Selected record comes from the LayerList panel
        auto* list = layerListPanel();
        if (list) {
            state.selected_record = list->selectedRecord();
            state.selected_index  = list->selectedIndex();
        }

        return state;
    }


    void drawStatusBar(const RenderState& state) {
        if (status_bar_row_ < 0) return;

        // Left side: focused panel's hint string
        PanelBase* focused = getPanel(focused_panel_);
        std::string hint = focused ? focused->statusHint() : "";

        // Right side: global indicators
        char right_buf[64];
        std::snprintf(right_buf, sizeof(right_buf),
                      " [%s] F%d  ?:help  q:quit",
                      paused_ ? "PAUSED" : "LIVE",
                      frame_count_ % 100);
        std::string right(right_buf);

        // Compose the full status line
        int avail = status_bar_cols_;
        int right_start = avail - static_cast<int>(right.size());

        // Clear the row
        move(status_bar_row_, 0);
        attron(A_REVERSE);
        for (int c = 0; c < avail; ++c) addch(' ');

        // Draw hint (left-aligned)
        if (!hint.empty()) {
            int hint_len = std::min(static_cast<int>(hint.size()),
                                     right_start > 0 ? right_start - 1 : avail);
            mvaddnstr(status_bar_row_, 0, hint.c_str(), hint_len);
        }

        // Draw indicators (right-aligned)
        if (right_start > 0 && right_start < avail) {
            mvaddstr(status_bar_row_, right_start, right.c_str());
        }

        attroff(A_REVERSE);
        refresh();   // stdscr refresh (status bar is on stdscr)
    }



    void drawHelpOverlay() {
        int rows, cols;
        getmaxyx(stdscr, rows, cols);

        static const char* kHelpLines[] = {
            "┌─────────────────── LLM Tracer Help ───────────────────────┐",
            "│                                                             │",
            "│  GLOBAL                                                     │",
            "│    Tab        Cycle panel focus                            │",
            "│    q / Q      Quit                                          │",
            "│    p / P      Pause / resume live updates                  │",
            "│    r / R      Reset ring buffer + latency stats            │",
            "│    ?          Toggle this help overlay                      │",
            "│                                                             │",
            "│  LAYERS PANEL (left)                                       │",
            "│    j / ↓      Move selection down                         │",
            "│    k / ↑      Move selection up                           │",
            "│    g / G      Jump to first / last layer                  │",
            "│    Space      Confirm selection                             │",
            "│    f          Toggle auto-follow (track newest layer)      │",
            "│                                                             │",
            "│  LATENCY PANEL (bottom-right)                              │",
            "│    t          Cycle sort mode (latency/order/name)        │",
            "│    m          Toggle last-pass vs moving average           │",
            "│                                                             │",
            "│  ATTENTION PANEL (middle-right)                            │",
            "│    [ / ]      Previous / next attention head               │",
            "│    t          Toggle single-head vs average-heads view     │",
            "│                                                             │",
            "│  Press ? to close this overlay                             │",
            "└─────────────────────────────────────────────────────────────┘",
        };

        const int nlines = static_cast<int>(sizeof(kHelpLines) / sizeof(kHelpLines[0]));
        int overlay_w = 63;
        int start_col = (cols - overlay_w) / 2;
        int start_row = (rows - nlines) / 2;
        start_col = std::max(0, start_col);
        start_row = std::max(0, start_row);

        attron(A_BOLD | COLOR_PAIR(PanelBase::colorPairTitle()));
        for (int i = 0; i < nlines && (start_row + i) < rows; ++i) {
            mvaddstr(start_row + i, start_col, kHelpLines[i]);
        }
        attroff(A_BOLD | COLOR_PAIR(PanelBase::colorPairTitle()));
    }



    void handleKey(const KeyEvent& key) {

        if (show_help_) {
            show_help_ = false;   // Any key closes help
            return;
        }


        if (key.isQuit())  { running_ = false; return; }
        if (key.isPause()) { paused_ = !paused_; return; }
        if (key.isHelp())  { show_help_ = true;  return; }

        if (key.isReset()) {
            reset_requested_.store(true, std::memory_order_relaxed);
            return;
        }


        if (key.isTab()) {
            cycleFocus();
            return;
        }


        PanelBase* focused = getPanel(focused_panel_);
        if (focused) focused->onKey(key);
    }

    void cycleFocus() {
        // Unfocus current
        PanelBase* current = getPanel(focused_panel_);
        if (current) current->setFocused(false);

        // Advance to next panel (wrap around)
        int next = (static_cast<int>(focused_panel_) + 1)
                 % static_cast<int>(PanelId::Count);
        focused_panel_ = static_cast<PanelId>(next);

        // Focus new panel
        PanelBase* next_panel = getPanel(focused_panel_);
        if (next_panel) next_panel->setFocused(true);
    }



    void handleResize() {
        // Let ncurses re-query the terminal size
        endwin();
        refresh();

        int rows, cols;
        getmaxyx(stdscr, rows, cols);

        // Recompute all panel rects and resize windows
        layoutPanels();

        // Force full redraw
        clearok(stdscr, TRUE);
        for (auto& panel : panels_) {
            if (panel) {
                panel->resize(panel->rect());
            }
        }
    }



    void shutdown() {
        for (auto& panel : panels_) {
            if (panel) panel->shutdown();
        }
        endwin();
        running_ = false;
    }



    PanelBase* getPanel(PanelId id) {
        int idx = static_cast<int>(id);
        if (idx < 0 || idx >= static_cast<int>(PanelId::Count)) return nullptr;
        return panels_[idx].get();
    }

    LayerListPanel* layerListPanel() {
        return static_cast<LayerListPanel*>(
            getPanel(PanelId::LayerList));
    }



    static void onSigwinch(int /*sig*/) {
        s_resize_flag_.store(true, std::memory_order_relaxed);
    }



    RingBufferView   buffer_view_;
    LatencyTracker&  latency_;
    TuiConfig        cfg_;

    // Panel array indexed by PanelId (LayerList=0, Metrics=1, Attention=2, Latency=3)
    std::array<std::unique_ptr<PanelBase>,
               static_cast<int>(PanelId::Count)> panels_;

    PanelId          focused_panel_;
    bool             paused_;
    std::atomic<bool> running_;
    std::atomic<bool> resize_pending_;
    bool             show_help_;
    uint64_t         frame_count_;

    int              status_bar_row_  = 0;
    int              status_bar_cols_ = 0;

    // Signals to the main() loop
    std::atomic<bool> reset_requested_{ false };

    // Global flag set by SIGWINCH handler
    static std::atomic<bool> s_resize_flag_;

public:
    // Expose reset flag for main() to poll and act on
    bool consumeResetRequest() {
        return reset_requested_.exchange(false);
    }
};

// Static member definition
inline std::atomic<bool> TuiApp::s_resize_flag_{ false };

} // namespace tui
} // namespace llmtracer