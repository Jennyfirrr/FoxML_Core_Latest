//! Service manager for systemd services

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::process::Command;
use std::time::Duration;

/// Service status
#[derive(Debug, Clone)]
pub enum ServiceStatus {
    Running,
    Stopped,
    Failed,
    Unknown,
}

/// Service manager
pub struct ServiceManager {
    service_name: String,
    status: ServiceStatus,
    status_text: String,
}

impl ServiceManager {
    pub fn new(service_name: String) -> Self {
        let status = Self::check_status(&service_name);
        let status_text = Self::get_status_text(&service_name);

        Self {
            service_name,
            status,
            status_text,
        }
    }

    /// Check if systemctl is available on this system
    fn has_systemctl() -> bool {
        Command::new("systemctl")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Check service status
    fn check_status(service_name: &str) -> ServiceStatus {
        if !Self::has_systemctl() {
            return ServiceStatus::Unknown;
        }

        let output = Command::new("systemctl")
            .args(&["is-active", service_name])
            .output();

        match output {
            Ok(output) => {
                let status_str = String::from_utf8_lossy(&output.stdout);
                let status = status_str.trim();
                match status {
                    "active" => ServiceStatus::Running,
                    "inactive" => ServiceStatus::Stopped,
                    "failed" => ServiceStatus::Failed,
                    _ => ServiceStatus::Unknown,
                }
            }
            Err(_) => ServiceStatus::Unknown,
        }
    }

    /// Get detailed status text
    fn get_status_text(service_name: &str) -> String {
        if !Self::has_systemctl() {
            return "systemctl not available on this system".to_string();
        }

        let output = Command::new("systemctl")
            .args(&["status", service_name, "--no-pager", "-l"])
            .output();

        match output {
            Ok(output) => {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .take(10)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            Err(_) => "Unable to get status".to_string(),
        }
    }

    /// Start service
    pub fn start(&mut self) -> Result<()> {
        Command::new("systemctl")
            .args(&["start", &self.service_name])
            .status()?;
        
        // Refresh status
        std::thread::sleep(Duration::from_millis(500));
        self.status = Self::check_status(&self.service_name);
        self.status_text = Self::get_status_text(&self.service_name);
        
        Ok(())
    }

    /// Stop service
    pub fn stop(&mut self) -> Result<()> {
        Command::new("systemctl")
            .args(&["stop", &self.service_name])
            .status()?;
        
        std::thread::sleep(Duration::from_millis(500));
        self.status = Self::check_status(&self.service_name);
        self.status_text = Self::get_status_text(&self.service_name);
        
        Ok(())
    }

    /// Restart service
    pub fn restart(&mut self) -> Result<()> {
        Command::new("systemctl")
            .args(&["restart", &self.service_name])
            .status()?;
        
        std::thread::sleep(Duration::from_millis(500));
        self.status = Self::check_status(&self.service_name);
        self.status_text = Self::get_status_text(&self.service_name);
        
        Ok(())
    }

    /// Refresh status
    pub fn refresh(&mut self) {
        self.status = Self::check_status(&self.service_name);
        self.status_text = Self::get_status_text(&self.service_name);
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let status_icon = match self.status {
            ServiceStatus::Running => "🟢",
            ServiceStatus::Stopped => "⚪",
            ServiceStatus::Failed => "🔴",
            ServiceStatus::Unknown => "❓",
        };
        
        let status_text = match self.status {
            ServiceStatus::Running => "Running",
            ServiceStatus::Stopped => "Stopped",
            ServiceStatus::Failed => "Failed",
            ServiceStatus::Unknown => "Unknown",
        };

        let block = Block::default()
            .title(format!("Service Manager: {}", self.service_name))
            .borders(Borders::ALL);

        let content = format!(
            "Status: {} {}\n\n{}\n\n[Start] [Stop] [Restart] [Refresh]",
            status_icon, status_text, self.status_text
        );

        let paragraph = Paragraph::new(content)
            .block(block)
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, area);

        Ok(())
    }
}
