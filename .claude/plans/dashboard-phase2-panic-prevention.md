# Phase 2: Rust TUI Panic Prevention

**Status**: Pending
**Parent**: `dashboard-bug-fixes-master.md`
**Scope**: 2 files, ~25 lines changed
**Depends on**: Nothing

## Problem

Two bugs can crash the Rust TUI or leave the terminal in a broken state:
1. If any code panics during `app.run()`, the terminal stays in raw mode with alternate screen active, requiring `reset` to recover.
2. If `run.progress > 1.0`, the progress bar rendering panics on unsigned integer underflow.

## Changes

### 2a. Terminal cleanup guard (`main.rs`, bug 7 — CRITICAL)

Add a RAII `TerminalGuard` struct whose `Drop` impl restores the terminal. This ensures cleanup even on panic.

```rust
/// Guard that restores terminal state on drop (including panic)
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
    }
}
```

In `main()`, create the guard after terminal setup and remove the explicit cleanup lines (the guard handles it):

```rust
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let _guard = TerminalGuard;

    let mut app = App::new().await?;
    app.run().await
}
```

### 2b. Safe progress bar rendering (`training.rs`, bug 8 — HIGH)

**Line 691-693** in `render_run_list()`: Clamp progress and use saturating subtraction.

```rust
// Before:
let progress_bar_width = 15;
let filled = (run.progress * progress_bar_width as f64) as usize;
let bar = format!("{}{}", "█".repeat(filled), "░".repeat(progress_bar_width - filled));

// After:
let progress_bar_width = 15usize;
let clamped = run.progress.clamp(0.0, 1.0);
let filled = (clamped * progress_bar_width as f64) as usize;
let empty = progress_bar_width.saturating_sub(filled);
let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
```

## Verification

- [ ] `cargo build --release` succeeds
- [ ] Terminal is restored if dashboard panics (manually test or inspect Drop impl)
- [ ] Progress bar renders correctly with `run.progress = 0.0, 0.5, 1.0, 1.5`
- [ ] No unsigned subtraction without `saturating_sub` in training.rs

## Files Changed

1. `DASHBOARD/dashboard/src/main.rs` — TerminalGuard struct + simplified main()
2. `DASHBOARD/dashboard/src/views/training.rs` — safe progress bar arithmetic
