//! Notification system
//!
//! Toast-style notifications that appear in the top-right corner.

use std::collections::VecDeque;
use std::io::Write;
use std::time::{Duration, Instant};

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Widget, Wrap};

use crate::themes::Theme;
use crate::ui::borders::Separators;

/// Notification severity level
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotificationLevel {
    /// Success - green, auto-dismiss after 3s
    Success,
    /// Warning - yellow, auto-dismiss after 5s
    Warning,
    /// Error - red, requires dismiss
    Error,
    /// Info - blue, auto-dismiss after 3s
    Info,
}

impl NotificationLevel {
    /// Get the default TTL for this level
    pub fn default_ttl(&self) -> Duration {
        match self {
            NotificationLevel::Success => Duration::from_secs(3),
            NotificationLevel::Info => Duration::from_secs(3),
            NotificationLevel::Warning => Duration::from_secs(5),
            NotificationLevel::Error => Duration::from_secs(60), // Effectively requires dismiss
        }
    }

    /// Get the icon for this level
    pub fn icon(&self) -> &'static str {
        match self {
            NotificationLevel::Success => Separators::CHECK,
            NotificationLevel::Warning => "⚠",
            NotificationLevel::Error => Separators::CROSS,
            NotificationLevel::Info => "ℹ",
        }
    }
}

/// A notification to display
#[derive(Clone, Debug)]
pub struct Notification {
    /// Notification severity
    pub level: NotificationLevel,
    /// Title/header
    pub title: String,
    /// Optional message body
    pub message: Option<String>,
    /// When the notification was created
    pub created_at: Instant,
    /// Time to live before auto-dismiss
    pub ttl: Duration,
    /// Whether this notification has been acknowledged
    pub acknowledged: bool,
}

impl Notification {
    /// Create a new notification
    pub fn new(level: NotificationLevel, title: impl Into<String>) -> Self {
        Self {
            level,
            title: title.into(),
            message: None,
            created_at: Instant::now(),
            ttl: level.default_ttl(),
            acknowledged: false,
        }
    }

    /// Add a message body
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Set a custom TTL
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Check if the notification should be removed
    pub fn is_expired(&self) -> bool {
        self.acknowledged || self.created_at.elapsed() > self.ttl
    }

    /// Acknowledge the notification
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }

    /// Success notification shortcut
    pub fn success(title: impl Into<String>) -> Self {
        Self::new(NotificationLevel::Success, title)
    }

    /// Warning notification shortcut
    pub fn warning(title: impl Into<String>) -> Self {
        Self::new(NotificationLevel::Warning, title)
    }

    /// Error notification shortcut
    pub fn error(title: impl Into<String>) -> Self {
        Self::new(NotificationLevel::Error, title)
    }

    /// Info notification shortcut
    pub fn info(title: impl Into<String>) -> Self {
        Self::new(NotificationLevel::Info, title)
    }
}

/// Notification manager - handles displaying and dismissing notifications
pub struct NotificationManager {
    notifications: VecDeque<Notification>,
    max_visible: usize,
    desktop_notifications_enabled: bool,
    sound_alerts_enabled: bool,
}

impl NotificationManager {
    /// Create a new notification manager
    pub fn new() -> Self {
        Self {
            notifications: VecDeque::new(),
            max_visible: 5,
            desktop_notifications_enabled: true,
            sound_alerts_enabled: true,
        }
    }

    /// Set maximum visible notifications
    pub fn max_visible(mut self, max: usize) -> Self {
        self.max_visible = max;
        self
    }

    /// Enable/disable desktop notifications
    pub fn desktop_notifications(mut self, enabled: bool) -> Self {
        self.desktop_notifications_enabled = enabled;
        self
    }

    /// Enable/disable sound alerts
    pub fn sound_alerts(mut self, enabled: bool) -> Self {
        self.sound_alerts_enabled = enabled;
        self
    }

    /// Add a notification
    pub fn push(&mut self, notification: Notification) {
        // Send desktop notification if enabled
        if self.desktop_notifications_enabled {
            self.send_desktop_notification(&notification);
        }

        // Play sound if enabled
        if self.sound_alerts_enabled {
            self.play_sound(&notification);
        }

        self.notifications.push_back(notification);

        // Limit total notifications
        while self.notifications.len() > 100 {
            self.notifications.pop_front();
        }
    }

    /// Remove expired notifications
    pub fn cleanup(&mut self) {
        self.notifications.retain(|n| !n.is_expired());
    }

    /// Dismiss the first notification
    pub fn dismiss_first(&mut self) {
        if let Some(notification) = self.notifications.front_mut() {
            notification.acknowledge();
        }
    }

    /// Dismiss all notifications
    pub fn dismiss_all(&mut self) {
        for notification in &mut self.notifications {
            notification.acknowledge();
        }
    }

    /// Get visible notifications
    pub fn visible(&self) -> impl Iterator<Item = &Notification> {
        self.notifications
            .iter()
            .filter(|n| !n.is_expired())
            .take(self.max_visible)
    }

    /// Check if there are any visible notifications
    pub fn has_notifications(&self) -> bool {
        self.notifications.iter().any(|n| !n.is_expired())
    }

    /// Send a desktop notification using notify-rust
    fn send_desktop_notification(&self, notification: &Notification) {
        #[cfg(feature = "desktop-notifications")]
        {
            use notify_rust::Notification as DesktopNotification;

            let urgency = match notification.level {
                NotificationLevel::Error => notify_rust::Urgency::Critical,
                NotificationLevel::Warning => notify_rust::Urgency::Normal,
                _ => notify_rust::Urgency::Low,
            };

            let _ = DesktopNotification::new()
                .summary(&format!("FOX ML: {}", notification.title))
                .body(notification.message.as_deref().unwrap_or(""))
                .icon("foxml")
                .urgency(urgency)
                .timeout(5000)
                .show();
        }
    }

    /// Play alert sound (terminal bell)
    fn play_sound(&self, notification: &Notification) {
        let bells = match notification.level {
            NotificationLevel::Error => 3,
            NotificationLevel::Warning => 2,
            _ => 1,
        };

        for _ in 0..bells {
            print!("\x07");
        }
        let _ = std::io::stdout().flush();
    }

    /// Render notifications in the given area (top-right corner)
    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let notifications: Vec<_> = self.visible().collect();
        if notifications.is_empty() {
            return;
        }

        // Calculate notification width (fixed)
        let notification_width = 40u16.min(area.width.saturating_sub(4));

        // Render each notification
        let mut y_offset = 1u16;
        for notification in notifications {
            let notification_height = if notification.message.is_some() { 4 } else { 3 };

            if y_offset + notification_height > area.height {
                break;
            }

            let notification_area = Rect {
                x: area.x.saturating_add(area.width.saturating_sub(notification_width + 2)),
                y: area.y + y_offset,
                width: notification_width,
                height: notification_height,
            };

            self.render_notification(notification, notification_area, buf, theme);
            y_offset += notification_height + 1;
        }
    }

    /// Render a single notification
    fn render_notification(
        &self,
        notification: &Notification,
        area: Rect,
        buf: &mut Buffer,
        theme: &Theme,
    ) {
        // Clear the area first
        Clear.render(area, buf);

        // Get the color for this level
        let border_color = match notification.level {
            NotificationLevel::Success => theme.success,
            NotificationLevel::Warning => theme.warning,
            NotificationLevel::Error => theme.error,
            NotificationLevel::Info => theme.info,
        };

        // Create the block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(border_color))
            .style(Style::default().bg(theme.surface_elevated));

        let inner = block.inner(area);
        block.render(area, buf);

        // Render title with icon
        let title = format!("{} {}", notification.level.icon(), notification.title);
        Paragraph::new(title)
            .style(Style::default().fg(theme.text_primary).bold())
            .render(
                Rect {
                    x: inner.x,
                    y: inner.y,
                    width: inner.width,
                    height: 1,
                },
                buf,
            );

        // Render message if present
        if let Some(message) = &notification.message {
            Paragraph::new(message.as_str())
                .style(Style::default().fg(theme.text_secondary))
                .wrap(Wrap { trim: true })
                .render(
                    Rect {
                        x: inner.x,
                        y: inner.y + 1,
                        width: inner.width,
                        height: inner.height.saturating_sub(1),
                    },
                    buf,
                );
        }
    }
}

impl Default for NotificationManager {
    fn default() -> Self {
        Self::new()
    }
}
