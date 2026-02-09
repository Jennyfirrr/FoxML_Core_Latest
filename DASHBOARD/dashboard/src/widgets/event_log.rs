//! Event log widget - displays streaming events from WebSocket

use ratatui::prelude::*;
use ratatui::widgets::*;
use std::collections::VecDeque;

use crate::api::events::TradingEvent;
use crate::themes::Theme;

/// Event log widget that displays streaming events
pub struct EventLog {
    events: VecDeque<TradingEvent>,
    max_events: usize,
}

impl EventLog {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_events),
            max_events,
        }
    }

    /// Push an event from WebSocket
    pub fn push_event(&mut self, event: TradingEvent) {
        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Add a simple string event (for backwards compatibility)
    pub fn add_event(&mut self, message: String) {
        let event = TradingEvent {
            event_type: "INFO".to_string(),
            timestamp: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
            severity: "info".to_string(),
            message,
            data: serde_json::Value::Null,
        };
        self.push_event(event);
    }

    /// Add event with severity
    pub fn add_event_with_severity(&mut self, severity: &str, message: String) {
        let event = TradingEvent {
            event_type: severity.to_uppercase(),
            timestamp: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
            severity: severity.to_lowercase(),
            message,
            data: serde_json::Value::Null,
        };
        self.push_event(event);
    }

    /// Get number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Render with theme support
    pub fn render_themed(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        // Render events in reverse order (newest first)
        let visible_height = area.height as usize;

        for (i, event) in self.events.iter().rev().take(visible_height).enumerate() {
            let y = area.y + i as u16;
            if y >= area.bottom() {
                break;
            }

            let severity_color = match event.severity_level() {
                "error" | "critical" => theme.error,
                "warning" => theme.warning,
                "info" => theme.accent,
                _ => theme.text_muted,
            };

            let timestamp = event.display_timestamp();
            let message = event.display_message();

            // Truncate message if too long
            let max_msg_len = (area.width as usize).saturating_sub(12); // [HH:MM:SS] + space
            let display_msg = if message.len() > max_msg_len {
                let truncated: String = message.chars().take(max_msg_len.saturating_sub(3)).collect();
                format!("{}...", truncated)
            } else {
                message
            };

            let line = Line::from(vec![
                Span::styled(
                    format!("[{}] ", timestamp),
                    Style::default().fg(theme.text_muted),
                ),
                Span::styled(
                    display_msg,
                    Style::default().fg(severity_color),
                ),
            ]);

            buf.set_line(area.x, y, &line, area.width);
        }
    }

    /// Render without theme (fallback)
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Event Log")
            .borders(Borders::ALL);

        let items: Vec<ListItem> = self
            .events
            .iter()
            .rev()
            .take((area.height as usize).saturating_sub(2))
            .map(|e| {
                let severity_color = match e.severity_level() {
                    "error" | "critical" => Color::Red,
                    "warning" => Color::Yellow,
                    "info" => Color::Cyan,
                    _ => Color::White,
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("[{}] ", e.display_timestamp()),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(
                        e.display_message(),
                        Style::default().fg(severity_color),
                    ),
                ]))
            })
            .collect();

        let list = List::new(items).block(block);
        use ratatui::widgets::Widget;
        Widget::render(list, area, buf);
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new(100)
    }
}
