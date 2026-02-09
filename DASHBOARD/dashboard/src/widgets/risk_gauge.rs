//! Risk gauge widget - displays risk metrics and warnings

use ratatui::prelude::*;

use crate::api::events::RiskStatus;
use crate::themes::Theme;

/// Risk gauge widget
pub struct RiskGauge {
    status: RiskStatus,
}

impl RiskGauge {
    pub fn new() -> Self {
        Self {
            status: RiskStatus::default(),
        }
    }

    /// Update risk status from API
    pub fn update(&mut self, status: RiskStatus) {
        self.status = status;
    }

    /// Check if kill switch is active
    pub fn is_kill_switch_active(&self) -> bool {
        self.status.kill_switch_active
    }

    /// Check if trading is allowed
    pub fn is_trading_allowed(&self) -> bool {
        self.status.trading_allowed
    }

    /// Render with theme support
    pub fn render_themed(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Kill switch status
                Constraint::Length(1), // Daily P&L
                Constraint::Length(1), // Drawdown
                Constraint::Length(1), // Exposure
                Constraint::Min(0),    // Warnings
            ])
            .split(area);

        // Kill switch status
        let (ks_text, ks_color) = if self.status.kill_switch_active {
            ("KILL SWITCH: ACTIVE", theme.error)
        } else if !self.status.trading_allowed {
            ("Trading: BLOCKED", theme.warning)
        } else {
            ("Trading: Allowed", theme.success)
        };
        buf.set_string(rows[0].x, rows[0].y, ks_text, Style::default().fg(ks_color).bold());

        // Daily P&L gauge
        self.render_gauge(
            buf,
            rows[1],
            "Daily P&L",
            self.status.daily_pnl_pct.abs(),
            self.status.daily_loss_limit_pct,
            self.status.daily_pnl_pct < 0.0,
            theme,
        );

        // Drawdown gauge
        self.render_gauge(
            buf,
            rows[2],
            "Drawdown",
            self.status.drawdown_pct,
            self.status.max_drawdown_limit_pct,
            true, // Drawdown is always "bad"
            theme,
        );

        // Exposure gauge
        self.render_gauge(
            buf,
            rows[3],
            "Exposure",
            self.status.gross_exposure,
            self.status.max_gross_exposure,
            false,
            theme,
        );

        // Warnings (if any)
        let warnings_start_y = rows[4].y;
        for (i, warning) in self.status.warnings.iter().take(3).enumerate() {
            let y = warnings_start_y + i as u16;
            if y >= area.bottom() {
                break;
            }

            let icon = match warning.severity.as_str() {
                "high" => "⚠",
                "medium" => "●",
                _ => "○",
            };

            let color = match warning.severity.as_str() {
                "high" => theme.error,
                "medium" => theme.warning,
                _ => theme.text_muted,
            };

            let msg = format!("{} {}", icon, warning.message);
            let truncated = if msg.len() > area.width as usize {
                let chars: String = msg.chars().take((area.width as usize).saturating_sub(3)).collect();
                format!("{}...", chars)
            } else {
                msg
            };

            buf.set_string(rows[4].x, y, &truncated, Style::default().fg(color));
        }
    }

    /// Render a gauge bar
    fn render_gauge(
        &self,
        buf: &mut Buffer,
        area: Rect,
        label: &str,
        value: f64,
        max: f64,
        is_negative: bool,
        theme: &Theme,
    ) {
        // Calculate percentage and color
        let pct = if max > 0.0 { (value / max).min(1.0) } else { 0.0 };
        let color = if pct > 0.8 {
            theme.error
        } else if pct > 0.5 {
            theme.warning
        } else {
            theme.success
        };

        // Format label and value
        let sign = if is_negative && value > 0.0 { "-" } else { "" };
        let text = format!(
            "{}: {}{:.1}% / {:.1}%",
            label, sign, value, max
        );

        // Render text
        buf.set_string(area.x, area.y, &text, Style::default().fg(color));

        // Render mini bar at the end if there's room
        let text_len = text.len() as u16;
        let bar_start = area.x + text_len + 2;
        let bar_width = area.width.saturating_sub(text_len + 3);

        if bar_width >= 5 {
            let filled = ((bar_width as f64 * pct) as u16).min(bar_width);
            let empty = bar_width - filled;

            // Draw filled portion
            buf.set_string(
                bar_start,
                area.y,
                "█".repeat(filled as usize),
                Style::default().fg(color),
            );
            // Draw empty portion
            buf.set_string(
                bar_start + filled,
                area.y,
                "░".repeat(empty as usize),
                Style::default().fg(theme.surface),
            );
        }
    }
}

impl Default for RiskGauge {
    fn default() -> Self {
        Self::new()
    }
}
