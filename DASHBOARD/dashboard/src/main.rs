//! FoxML Dashboard - Rust TUI for monitoring autonomous trading and training
//!
//! Entry point for the dashboard application.

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io;
use tracing_subscriber;

mod app;
mod api;
mod views;
mod widgets;
mod themes;
mod launcher;
mod ui;

use app::App;

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

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    // Guard ensures cleanup even on panic
    let _guard = TerminalGuard;

    // Create and run app
    let mut app = App::new().await?;
    let result = app.run().await;

    // Explicit cleanup (guard also cleans up, but this gives us error propagation)
    drop(_guard);

    result
}
