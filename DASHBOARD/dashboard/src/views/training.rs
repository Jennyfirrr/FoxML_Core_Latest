//! Training pipeline view - training progress monitoring
//!
//! Monitors local training runs by reading the events file directly.
//! No WebSocket needed since training runs locally.

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use walkdir::WalkDir;
use regex::Regex;

use crate::api::events::TrainingEvent;
use crate::themes::Theme;
use crate::ui::borders::Separators;
use crate::ui::panels::Panel;

/// Path to training events file (written by Python TrainingEventEmitter)
const TRAINING_EVENTS_FILE: &str = "/tmp/foxml_training_events.jsonl";

/// Path to PID file for process detection
const TRAINING_PID_FILE: &str = "/tmp/foxml_training.pid";

/// Information from PID file
struct PidInfo {
    #[allow(dead_code)]
    pid: u32,
    run_id: String,
    is_alive: bool,
}

/// Training run info
#[derive(Debug, Clone)]
pub struct TrainingRun {
    run_id: String,
    path: PathBuf,
    status: String,
    progress: f64,
    stage: String,
    current_target: Option<String>,
    targets_complete: i64,
    targets_total: i64,
}

/// Training view - training pipeline monitoring
pub struct TrainingView {
    runs: Vec<TrainingRun>,
    selected: usize,
    last_scan: std::time::Instant,
    theme: Theme,
    // File-based event monitoring
    event_file_pos: u64,
    last_event_poll: std::time::Instant,
    // Current live run info (from events file)
    live_run_id: Option<String>,
    live_stage: String,
    live_progress: f64,
    live_current_target: Option<String>,
    live_targets_complete: i64,
    live_targets_total: i64,
    live_message: Option<String>,
    is_monitoring: bool,
}

impl TrainingView {
    pub fn new() -> Self {
        let mut view = Self {
            runs: Vec::new(),
            selected: 0,
            last_scan: std::time::Instant::now(),
            theme: Theme::load(),
            event_file_pos: 0,
            last_event_poll: std::time::Instant::now(),
            live_run_id: None,
            live_stage: "idle".to_string(),
            live_progress: 0.0,
            live_current_target: None,
            live_targets_complete: 0,
            live_targets_total: 0,
            live_message: None,
            is_monitoring: false,
        };
        // Auto-scan on initialization
        let _ = view.scan_runs();
        // Start monitoring events file
        view.start_monitoring();
        view
    }

    /// Start monitoring the events file
    pub fn start_monitoring(&mut self) {
        self.is_monitoring = true;

        // On startup, read existing events to recover current state
        // This allows showing progress if a run is already in progress
        self.recover_current_state();
    }

    /// Recover current training state from existing events file
    fn recover_current_state(&mut self) {
        let file = match fs::File::open(TRAINING_EVENTS_FILE) {
            Ok(f) => f,
            Err(_) => {
                self.event_file_pos = 0;
                return;
            }
        };

        // Read all events to find the most recent run
        let reader = BufReader::new(&file);
        let mut last_run_id: Option<String> = None;
        let mut last_run_events: Vec<TrainingEvent> = Vec::new();
        let mut run_completed = false;

        for line in reader.lines() {
            if let Ok(line_str) = line {
                let trimmed = line_str.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<TrainingEvent>(trimmed) {
                    // Check if this is a new run
                    if !event.run_id.is_empty() {
                        if last_run_id.as_ref() != Some(&event.run_id) {
                            // New run started - reset events
                            last_run_id = Some(event.run_id.clone());
                            last_run_events.clear();
                            run_completed = false;
                        }
                    }

                    // Track if run completed
                    if event.event_type == "run_complete" {
                        run_completed = true;
                    }

                    last_run_events.push(event);
                }
            }
        }

        // Set file position to end for future polling
        if let Ok(metadata) = file.metadata() {
            self.event_file_pos = metadata.len();
        }

        // If the last run is NOT completed, replay its events to set current state
        if !run_completed && !last_run_events.is_empty() {
            for event in last_run_events {
                self.handle_training_event(event);
            }
        }
    }

    /// Poll for new training events from the events file
    pub fn poll_events(&mut self) {
        if !self.is_monitoring {
            return;
        }

        // Only poll every 500ms to avoid excessive file reads
        if self.last_event_poll.elapsed().as_millis() < 500 {
            return;
        }
        self.last_event_poll = std::time::Instant::now();

        // Read new events from file
        let events = self.read_new_events();
        for event in events {
            self.handle_training_event(event);
        }
    }

    /// Read new events from the JSONL file since last position
    fn read_new_events(&mut self) -> Vec<TrainingEvent> {
        let mut events = Vec::new();

        let file = match fs::File::open(TRAINING_EVENTS_FILE) {
            Ok(f) => f,
            Err(_) => return events, // File doesn't exist yet
        };

        let mut reader = BufReader::new(file);

        // Seek to last position
        if reader.seek(SeekFrom::Start(self.event_file_pos)).is_err() {
            return events;
        }

        // Read new lines
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                if let Ok(event) = serde_json::from_str::<TrainingEvent>(trimmed) {
                    events.push(event);
                }
            }
            line.clear();
        }

        // Update position
        if let Ok(pos) = reader.stream_position() {
            self.event_file_pos = pos;
        }

        events
    }

    /// Handle incoming training event
    fn handle_training_event(&mut self, event: TrainingEvent) {
        // Update live state
        if !event.run_id.is_empty() {
            self.live_run_id = Some(event.run_id.clone());
        }

        match event.event_type.as_str() {
            "progress" => {
                self.live_stage = event.effective_stage().to_string();
                self.live_progress = event.progress_pct;
                self.live_current_target = event.current_target.clone();
                self.live_targets_complete = event.targets_complete;
                self.live_targets_total = event.targets_total;
                self.live_message = event.message.clone();
            }
            "stage_change" => {
                self.live_stage = event.effective_stage().to_string();
            }
            "target_start" => {
                self.live_current_target = event.target.clone();
            }
            "target_complete" => {
                self.live_targets_complete = event.targets_complete;
            }
            "run_complete" => {
                // Clear live state on completion
                self.live_stage = "completed".to_string();
                self.live_progress = 100.0;
                self.live_current_target = None;
                self.live_message = event.message.clone();
                // Trigger a rescan to update the run list
                let _ = self.scan_runs();
            }
            "error" => {
                self.live_message = event.error_message.clone();
            }
            _ => {}
        }

        // Update the matching run in the list if we have a run_id
        if let Some(run_id) = &self.live_run_id {
            for run in &mut self.runs {
                if run.run_id == *run_id {
                    run.progress = self.live_progress / 100.0;
                    run.stage = self.live_stage.clone();
                    run.current_target = self.live_current_target.clone();
                    run.targets_complete = self.live_targets_complete;
                    run.targets_total = self.live_targets_total;
                    run.status = if self.live_stage == "completed" {
                        "completed".to_string()
                    } else {
                        "running".to_string()
                    };
                    break;
                }
            }
        }
    }

    /// Scan for training runs
    pub fn scan_runs(&mut self) -> Result<()> {
        let mut runs = Vec::new();

        // Get project root - try multiple strategies
        let project_root = Self::find_project_root()?;

        // Scan RESULTS directory recursively - find ALL manifest.json files anywhere in RESULTS
        let results_dir = project_root.join("RESULTS");

        // Also check TRAINING/results as fallback
        let training_results_dir = project_root.join("TRAINING/results");

        // Detect running training processes
        let running_runs = Self::detect_running_processes()?;

        // Scan RESULTS recursively (no depth limit - find everything)
        if results_dir.exists() {
            for entry in WalkDir::new(&results_dir)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some("manifest.json") {
                    // Found a manifest - the run directory is the parent
                    if let Some(run_dir) = path.parent() {
                        // Use full directory path as run_id, or directory name if available
                        let run_id = if let Some(dir_name) = run_dir.file_name().and_then(|n| n.to_str()) {
                            dir_name.to_string()
                        } else {
                            run_dir.to_string_lossy().to_string()
                        };

                        // Skip if we already found this run (by path)
                        let run_path = run_dir.to_path_buf();
                        if runs.iter().any(|r: &TrainingRun| r.path == run_path) {
                            continue;
                        }

                        // Check if this run is currently running
                        let is_running = running_runs.iter().any(|r| r == run_dir);

                        // Try to read progress and status from manifest
                        let (progress, status, stage) = if let Ok(content) = fs::read_to_string(path) {
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                                let progress_val = json.get("progress")
                                    .or_else(|| json.get("completion_percentage"))
                                    .and_then(|p| p.as_f64())
                                    .unwrap_or(0.0);

                                let status_val = if is_running {
                                    "running".to_string()
                                } else {
                                    json.get("status")
                                        .or_else(|| json.get("run_status"))
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("completed")
                                        .to_string()
                                };

                                let stage_val = json.get("current_stage")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("idle")
                                    .to_string();

                                (progress_val, status_val, stage_val)
                            } else {
                                (if is_running { 0.5 } else { 0.0 },
                                 if is_running { "running".to_string() } else { "completed".to_string() },
                                 "idle".to_string())
                            }
                        } else {
                            (if is_running { 0.5 } else { 0.0 },
                             if is_running { "running".to_string() } else { "unknown".to_string() },
                             "idle".to_string())
                        };

                        runs.push(TrainingRun {
                            run_id,
                            path: run_path,
                            status,
                            progress,
                            stage,
                            current_target: None,
                            targets_complete: 0,
                            targets_total: 0,
                        });
                    }
                }
            }
        }

        // Also scan TRAINING/results if it exists
        if training_results_dir.exists() {
            for entry in WalkDir::new(&training_results_dir)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some("manifest.json") {
                    if let Some(run_dir) = path.parent() {
                        let run_id = if let Some(dir_name) = run_dir.file_name().and_then(|n| n.to_str()) {
                            dir_name.to_string()
                        } else {
                            run_dir.to_string_lossy().to_string()
                        };

                        let run_path = run_dir.to_path_buf();
                        if runs.iter().any(|r: &TrainingRun| r.path == run_path) {
                            continue;
                        }

                        let is_running = running_runs.iter().any(|r| r == run_dir);

                        let (progress, status, stage) = if let Ok(content) = fs::read_to_string(path) {
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                                let progress_val = json.get("progress")
                                    .or_else(|| json.get("completion_percentage"))
                                    .and_then(|p| p.as_f64())
                                    .unwrap_or(0.0);

                                let status_val = if is_running {
                                    "running".to_string()
                                } else {
                                    json.get("status")
                                        .or_else(|| json.get("run_status"))
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("completed")
                                        .to_string()
                                };

                                let stage_val = json.get("current_stage")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("idle")
                                    .to_string();

                                (progress_val, status_val, stage_val)
                            } else {
                                (if is_running { 0.5 } else { 0.0 },
                                 if is_running { "running".to_string() } else { "completed".to_string() },
                                 "idle".to_string())
                            }
                        } else {
                            (if is_running { 0.5 } else { 0.0 },
                             if is_running { "running".to_string() } else { "unknown".to_string() },
                             "idle".to_string())
                        };

                        runs.push(TrainingRun {
                            run_id,
                            path: run_path,
                            status,
                            progress,
                            stage,
                            current_target: None,
                            targets_complete: 0,
                            targets_total: 0,
                        });
                    }
                }
            }
        }

        // Sort by path (newest first, assuming newer runs have later timestamps in path)
        runs.sort_by(|a, b| {
            // Try to extract timestamp from path for better sorting
            let a_ts = Self::extract_timestamp_from_path(&a.path);
            let b_ts = Self::extract_timestamp_from_path(&b.path);
            b_ts.cmp(&a_ts).then_with(|| b.path.cmp(&a.path))
        });
        self.runs = runs;
        self.last_scan = std::time::Instant::now();
        Ok(())
    }

    /// Detect currently running training processes
    fn detect_running_processes() -> Result<Vec<PathBuf>> {
        let mut running_dirs = Vec::new();

        // First, try reading from PID file (more reliable)
        if let Some(run_info) = Self::read_pid_file() {
            // Verify the process is still running
            if run_info.is_alive {
                // Try to get output dir from the events file
                if let Ok(content) = fs::read_to_string(TRAINING_EVENTS_FILE) {
                    // Find events for this run_id
                    for line in content.lines().rev() {
                        if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
                            if event.get("run_id").and_then(|v| v.as_str()) == Some(&run_info.run_id) {
                                // This run is active - we can't determine output dir from events
                                // but we know training is running
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: Check for intelligent_trainer.py processes via ps
        if running_dirs.is_empty() {
            if let Ok(output) = std::process::Command::new("ps")
                .args(["aux"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    // Look for intelligent_trainer.py with --output-dir
                    if line.contains("intelligent_trainer.py") && line.contains("--output-dir") {
                        // Extract output directory from command line
                        if let Some(output_dir_start) = line.find("--output-dir") {
                            let after_flag = &line[output_dir_start + 12..]; // Skip "--output-dir"
                            let parts: Vec<&str> = after_flag.split_whitespace().collect();
                            if let Some(dir_str) = parts.first() {
                                let dir_path = PathBuf::from(dir_str);
                                if dir_path.exists() {
                                    running_dirs.push(dir_path);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(running_dirs)
    }

    /// Read PID file and check if process is alive
    fn read_pid_file() -> Option<PidInfo> {
        let content = fs::read_to_string(TRAINING_PID_FILE).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        let pid = json.get("pid")?.as_u64()? as u32;
        let run_id = json.get("run_id")?.as_str()?.to_string();

        // Check if process is still alive by checking /proc/{pid}
        let proc_path = PathBuf::from(format!("/proc/{}", pid));
        let is_alive = proc_path.exists();

        Some(PidInfo { pid, run_id, is_alive })
    }

    /// Find project root by looking for RESULTS directory
    fn find_project_root() -> Result<PathBuf> {
        // Start from current directory
        let mut current = std::env::current_dir()?;

        // Walk up the directory tree looking for RESULTS
        loop {
            let results = current.join("RESULTS");
            if results.exists() && results.is_dir() {
                return Ok(current);
            }

            // Try parent directory
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                // Reached root, fallback to current_dir
                return Ok(std::env::current_dir()?);
            }
        }
    }

    /// Extract timestamp from path for sorting (try to find YYYYMMDD_HHMMSS pattern)
    fn extract_timestamp_from_path(path: &PathBuf) -> String {
        // Look for timestamp patterns in path components
        let path_str = path.to_string_lossy();

        // Try to find YYYYMMDD_HHMMSS pattern
        if let Ok(re) = Regex::new(r"(\d{8}_\d{6})") {
            if let Some(captures) = re.captures(&path_str) {
                if let Some(m) = captures.get(1) {
                    return m.as_str().to_string();
                }
            }
        }

        // Fallback: use full path for sorting
        path_str.to_string()
    }

    /// Render header with monitoring status
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let status_indicator = if self.is_monitoring && self.live_run_id.is_some() {
            Span::styled(
                format!("{} Live", Separators::CIRCLE_FILLED),
                Style::default().fg(self.theme.success),
            )
        } else if self.is_monitoring {
            Span::styled(
                format!("{} Monitoring", Separators::CIRCLE_FILLED),
                Style::default().fg(self.theme.accent),
            )
        } else {
            Span::styled(
                format!("{} Idle", Separators::CIRCLE_EMPTY),
                Style::default().fg(self.theme.text_muted),
            )
        };

        let title = Line::from(vec![
            Span::styled(
                format!("{} ", Separators::DIAMOND),
                Style::default().fg(self.theme.accent),
            ),
            Span::styled(
                "Training Pipeline Monitor",
                Style::default().fg(self.theme.text_primary).bold(),
            ),
            Span::styled(
                "  │  ",
                Style::default().fg(self.theme.border),
            ),
            status_indicator,
        ]);
        frame.render_widget(Paragraph::new(title), area);
    }

    /// Render live progress panel
    fn render_live_progress(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Live Progress").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.live_run_id.is_none() {
            let text = Paragraph::new("No live training in progress.\nStart a training run to see live progress here.")
                .style(Style::default().fg(self.theme.text_muted));
            frame.render_widget(text, inner);
            return;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Stage
                Constraint::Length(1), // Progress bar
                Constraint::Length(1), // Target info
                Constraint::Length(1), // Message
            ])
            .split(inner);

        // Stage
        let stage_color = match self.live_stage.as_str() {
            "ranking" => self.theme.warning,
            "feature_selection" => self.theme.accent,
            "training" => self.theme.success,
            "completed" => self.theme.success,
            _ => self.theme.text_muted,
        };
        let stage_line = Line::from(vec![
            Span::styled("Stage: ", Style::default().fg(self.theme.text_muted)),
            Span::styled(&self.live_stage, Style::default().fg(stage_color).bold()),
        ]);
        frame.render_widget(Paragraph::new(stage_line), rows[0]);

        // Progress bar
        let progress_bar_width = rows[1].width.saturating_sub(10) as usize;
        let filled = ((self.live_progress / 100.0) * progress_bar_width as f64) as usize;
        let empty = progress_bar_width.saturating_sub(filled);
        let bar = format!(
            "[{}{}] {:>5.1}%",
            "█".repeat(filled),
            "░".repeat(empty),
            self.live_progress
        );
        let progress_line = Paragraph::new(bar)
            .style(Style::default().fg(self.theme.accent));
        frame.render_widget(progress_line, rows[1]);

        // Target info
        let target_info = if let Some(target) = &self.live_current_target {
            format!("Target: {} ({}/{})", target, self.live_targets_complete, self.live_targets_total)
        } else if self.live_targets_total > 0 {
            format!("Targets: {}/{}", self.live_targets_complete, self.live_targets_total)
        } else {
            "".to_string()
        };
        let target_line = Paragraph::new(target_info)
            .style(Style::default().fg(self.theme.text_secondary));
        frame.render_widget(target_line, rows[2]);

        // Message
        if let Some(msg) = &self.live_message {
            let msg_line = Paragraph::new(msg.as_str())
                .style(Style::default().fg(self.theme.text_muted));
            frame.render_widget(msg_line, rows[3]);
        }
    }

    /// Render run list
    fn render_run_list(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Training Runs").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.runs.is_empty() {
            let text = Paragraph::new("No training runs found.\nPress [r] to scan RESULTS/ directory.")
                .style(Style::default().fg(self.theme.text_muted));
            frame.render_widget(text, inner);
            return;
        }

        let items: Vec<ListItem> = self
            .runs
            .iter()
            .enumerate()
            .map(|(i, run)| {
                let is_selected = i == self.selected;
                let indicator = if is_selected { "▸" } else { " " };

                let status_color = match run.status.as_str() {
                    "running" => self.theme.warning,
                    "completed" => self.theme.success,
                    "failed" => self.theme.error,
                    _ => self.theme.text_muted,
                };

                let progress_bar_width = 15usize;
                let clamped = run.progress.clamp(0.0, 1.0);
                let filled = (clamped * progress_bar_width as f64) as usize;
                let empty = progress_bar_width.saturating_sub(filled);
                let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

                let line = Line::from(vec![
                    Span::styled(indicator, Style::default().fg(if is_selected { self.theme.accent } else { self.theme.text_muted })),
                    Span::styled(" ", Style::default()),
                    Span::styled(&run.run_id, Style::default().fg(if is_selected { self.theme.accent } else { self.theme.text_primary }).bold()),
                    Span::styled(" ", Style::default()),
                    Span::styled(&run.status, Style::default().fg(status_color)),
                    Span::styled(" [", Style::default().fg(self.theme.border)),
                    Span::styled(bar, Style::default().fg(self.theme.accent)),
                    Span::styled("]", Style::default().fg(self.theme.border)),
                    Span::styled(format!(" {:>5.0}%", run.progress * 100.0), Style::default().fg(self.theme.text_secondary)),
                ]);

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render footer with keybindings
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = vec![
            ("↑↓/jk", "Navigate"),
            ("r", "Refresh"),
            ("c", "Clear events"),
            ("b/Esc", "Back"),
        ];

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            }
            spans.push(Span::styled(
                format!("[{}]", key),
                Style::default().fg(self.theme.accent),
            ));
            spans.push(Span::styled(
                format!(" {}", desc),
                Style::default().fg(self.theme.text_secondary),
            ));
        }

        // Add last scan time
        spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
        spans.push(Span::styled(
            format!("Scanned: {:.1}s ago", self.last_scan.elapsed().as_secs_f64()),
            Style::default().fg(self.theme.text_muted),
        ));

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for TrainingView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for TrainingView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Poll for events (non-blocking)
        self.poll_events();

        // Clear background with theme color
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Length(6),  // Live progress
                Constraint::Min(10),    // Run list
                Constraint::Length(2),  // Footer
            ])
            .margin(1)
            .split(area);

        // Header with title
        self.render_header(frame, chunks[0]);

        // Live progress panel
        self.render_live_progress(frame, chunks[1]);

        // Run list
        self.render_run_list(frame, chunks[2]);

        // Footer
        self.render_footer(frame, chunks[3]);

        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyCode) -> Result<bool> {
        match key {
            crossterm::event::KeyCode::Char('r') => {
                self.scan_runs()?;
                Ok(false)
            }
            crossterm::event::KeyCode::Char('c') => {
                // Clear live state and reset file position to end
                self.live_run_id = None;
                self.live_stage = "idle".to_string();
                self.live_progress = 0.0;
                self.live_current_target = None;
                self.live_targets_complete = 0;
                self.live_targets_total = 0;
                self.live_message = None;
                // Reset to end of file
                if let Ok(file) = fs::File::open(TRAINING_EVENTS_FILE) {
                    if let Ok(metadata) = file.metadata() {
                        self.event_file_pos = metadata.len();
                    }
                }
                Ok(false)
            }
            crossterm::event::KeyCode::Up | crossterm::event::KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                Ok(false)
            }
            crossterm::event::KeyCode::Down | crossterm::event::KeyCode::Char('j') => {
                if self.selected < self.runs.len().saturating_sub(1) {
                    self.selected += 1;
                }
                Ok(false)
            }
            crossterm::event::KeyCode::Char('g') => {
                // Go to top (gg)
                self.selected = 0;
                Ok(false)
            }
            crossterm::event::KeyCode::Char('G') => {
                // Go to bottom (G)
                if !self.runs.is_empty() {
                    self.selected = self.runs.len() - 1;
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }
}
