//! Model Selector View - browse and select trained models for live trading
//!
//! Scans RESULTS/runs/ for completed training runs and allows selecting
//! which model set to use for LIVE_TRADING.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Training run metadata from manifest.json
#[derive(Debug, Clone, Deserialize)]
struct RunManifest {
    run_id: String,
    #[serde(default)]
    experiment_name: Option<String>,
    #[serde(default)]
    start_time: Option<String>,
    #[serde(default)]
    end_time: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    git_sha: Option<String>,
    #[serde(default)]
    config_fingerprint: Option<String>,
    #[serde(default)]
    target_index: Option<HashMap<String, TargetEntry>>,
}

/// Target entry from manifest
#[derive(Debug, Clone, Deserialize)]
struct TargetEntry {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    best_model_family: Option<String>,
    #[serde(default)]
    best_auc: Option<f64>,
}

/// Processed run info for display
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub run_id: String,
    pub experiment: String,
    pub date: String,
    pub status: String,
    pub target_count: usize,
    pub completed_targets: usize,
    pub best_auc: Option<f64>,
    pub avg_auc: Option<f64>,
    pub git_sha: String,
    pub path: PathBuf,
    pub targets: Vec<TargetInfo>,
}

/// Target info for detail view
#[derive(Debug, Clone)]
pub struct TargetInfo {
    pub name: String,
    pub status: String,
    pub best_family: String,
    pub auc: Option<f64>,
}

/// Model selector view
pub struct ModelSelectorView {
    theme: Theme,
    runs: Vec<RunInfo>,
    selected: usize,
    scroll: usize,
    detail_scroll: usize,
    show_detail: bool,
    active_run: Option<String>,
    message: Option<(String, bool)>, // (message, is_error)
}

impl ModelSelectorView {
    pub fn new() -> Self {
        let mut view = Self {
            theme: Theme::load(),
            runs: Vec::new(),
            selected: 0,
            scroll: 0,
            detail_scroll: 0,
            show_detail: false,
            active_run: None,
            message: None,
        };
        view.scan_runs();
        view.load_active_run();
        view
    }

    /// Scan RESULTS/ for training runs (all subdirectories)
    fn scan_runs(&mut self) {
        let results_dir = PathBuf::from("RESULTS");
        if !results_dir.exists() {
            self.runs = Vec::new();
            return;
        }

        let mut runs = Vec::new();

        // Scan all subdirectories in RESULTS/
        if let Ok(subdirs) = fs::read_dir(&results_dir) {
            for subdir_entry in subdirs.filter_map(|e| e.ok()) {
                let subdir_path = subdir_entry.path();
                if !subdir_path.is_dir() {
                    continue;
                }

                let subdir_name = subdir_entry
                    .file_name()
                    .to_string_lossy()
                    .to_string();

                // Check if this directory has a manifest.json (it's a run directory)
                let direct_manifest = subdir_path.join("manifest.json");
                if direct_manifest.exists() {
                    // This subdirectory IS a run directory
                    if let Ok(content) = fs::read_to_string(&direct_manifest) {
                        if let Ok(manifest) = serde_json::from_str::<RunManifest>(&content) {
                            let run_info = Self::process_manifest(manifest, subdir_path.clone());
                            runs.push(run_info);
                        }
                    }
                    continue;
                }

                // Otherwise, scan subdirectory for run directories
                if let Ok(entries) = fs::read_dir(&subdir_path) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if !path.is_dir() {
                            continue;
                        }

                        let manifest_path = path.join("manifest.json");
                        if !manifest_path.exists() {
                            continue;
                        }

                        if let Ok(content) = fs::read_to_string(&manifest_path) {
                            if let Ok(manifest) = serde_json::from_str::<RunManifest>(&content) {
                                let mut run_info = Self::process_manifest(manifest, path);
                                // Add folder tag
                                run_info.experiment = format!("{} ({})", run_info.experiment, subdir_name);
                                runs.push(run_info);
                            }
                        }
                    }
                }
            }
        }

        // Sort by date (newest first) - run_id typically starts with date
        runs.sort_by(|a, b| b.run_id.cmp(&a.run_id));

        self.runs = runs;
    }

    /// Process manifest into RunInfo
    fn process_manifest(manifest: RunManifest, path: PathBuf) -> RunInfo {
        let mut targets = Vec::new();
        let mut best_auc: Option<f64> = None;
        let mut auc_sum = 0.0;
        let mut auc_count = 0;
        let mut completed = 0;

        if let Some(ref target_index) = manifest.target_index {
            for (name, entry) in target_index {
                let status = entry.status.clone().unwrap_or_else(|| "unknown".to_string());
                if status == "completed" || status == "trained" {
                    completed += 1;
                }

                if let Some(auc) = entry.best_auc {
                    auc_sum += auc;
                    auc_count += 1;
                    if best_auc.map_or(true, |b| auc > b) {
                        best_auc = Some(auc);
                    }
                }

                targets.push(TargetInfo {
                    name: name.clone(),
                    status,
                    best_family: entry.best_model_family.clone().unwrap_or_else(|| "-".to_string()),
                    auc: entry.best_auc,
                });
            }
        }

        // Sort targets by name
        targets.sort_by(|a, b| a.name.cmp(&b.name));

        let avg_auc = if auc_count > 0 {
            Some(auc_sum / auc_count as f64)
        } else {
            None
        };

        // Extract date from run_id or start_time
        let date = manifest
            .start_time
            .as_ref()
            .map(|s| s.split('T').next().unwrap_or(s).to_string())
            .or_else(|| {
                // Try to extract date from run_id (format: YYYYMMDD_HHMMSS_...)
                if manifest.run_id.len() >= 8 {
                    let d = &manifest.run_id[..8];
                    if d.chars().all(|c| c.is_ascii_digit()) {
                        Some(format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..8]))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "Unknown".to_string());

        RunInfo {
            run_id: manifest.run_id,
            experiment: manifest.experiment_name.unwrap_or_else(|| "default".to_string()),
            date,
            status: manifest.status.unwrap_or_else(|| "unknown".to_string()),
            target_count: manifest.target_index.as_ref().map(|t| t.len()).unwrap_or(0),
            completed_targets: completed,
            best_auc,
            avg_auc,
            git_sha: manifest.git_sha.map(|s| s[..7.min(s.len())].to_string()).unwrap_or_else(|| "-".to_string()),
            path,
            targets,
        }
    }

    /// Load currently active run from LIVE_TRADING config
    fn load_active_run(&mut self) {
        // Check for active_model symlink or config
        let active_path = PathBuf::from("LIVE_TRADING/models/active");
        if active_path.exists() {
            if let Ok(target) = fs::read_link(&active_path) {
                if let Some(name) = target.file_name() {
                    self.active_run = Some(name.to_string_lossy().to_string());
                }
            }
        }

        // Also check config file
        let config_path = PathBuf::from("LIVE_TRADING/config/model_config.yaml");
        if config_path.exists() {
            if let Ok(content) = fs::read_to_string(&config_path) {
                for line in content.lines() {
                    if line.starts_with("run_id:") || line.starts_with("active_run:") {
                        if let Some(value) = line.split(':').nth(1) {
                            self.active_run = Some(value.trim().trim_matches('"').to_string());
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Set the selected run as active for live trading
    fn set_active_run(&mut self) -> Result<String> {
        if self.runs.is_empty() {
            return Ok("No runs available".to_string());
        }

        if self.selected >= self.runs.len() {
            self.selected = self.runs.len().saturating_sub(1);
        }
        let run = &self.runs[self.selected];

        // Create LIVE_TRADING/models directory if needed
        let models_dir = PathBuf::from("LIVE_TRADING/models");
        fs::create_dir_all(&models_dir)?;

        // Create/update symlink
        let active_link = models_dir.join("active");
        if active_link.exists() {
            fs::remove_file(&active_link)?;
        }

        #[cfg(unix)]
        std::os::unix::fs::symlink(&run.path, &active_link)?;

        self.active_run = Some(run.run_id.clone());

        Ok(format!("Set {} as active model", run.run_id))
    }

    /// Render run list
    fn render_list(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Training Runs").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.runs.is_empty() {
            let text = Paragraph::new("No training runs found in RESULTS/runs/")
                .style(Style::default().fg(self.theme.text_muted))
                .alignment(Alignment::Center);
            frame.render_widget(text, inner);
            return;
        }

        let visible_height = inner.height as usize;
        let start = self.scroll;
        let end = (start + visible_height).min(self.runs.len());

        let items: Vec<ListItem> = self.runs[start..end]
            .iter()
            .enumerate()
            .map(|(i, run)| {
                let idx = start + i;
                let is_selected = idx == self.selected;
                let is_active = self.active_run.as_ref().map_or(false, |a| a == &run.run_id);

                // Status indicator
                let status_color = match run.status.as_str() {
                    "completed" => self.theme.success,
                    "failed" => self.theme.error,
                    "running" => self.theme.warning,
                    _ => self.theme.text_muted,
                };

                // AUC display
                let auc_text = run.best_auc.map_or("-".to_string(), |a| format!("{:.4}", a));

                let active_marker = if is_active { " [ACTIVE]" } else { "" };

                let content = Line::from(vec![
                    Span::styled(
                        if is_selected { ">" } else { " " },
                        Style::default().fg(self.theme.accent),
                    ),
                    Span::styled(
                        format!(" {:20}", &run.run_id[..20.min(run.run_id.len())]),
                        Style::default().fg(if is_selected {
                            self.theme.accent
                        } else {
                            self.theme.text_primary
                        }),
                    ),
                    Span::styled(
                        format!(" {:10}", run.date),
                        Style::default().fg(self.theme.text_secondary),
                    ),
                    Span::styled(
                        format!(" {:>3}/{:<3}", run.completed_targets, run.target_count),
                        Style::default().fg(self.theme.text_muted),
                    ),
                    Span::styled(
                        format!(" {:>7}", auc_text),
                        Style::default().fg(self.theme.accent),
                    ),
                    Span::styled(
                        format!(" {:9}", run.status),
                        Style::default().fg(status_color),
                    ),
                    Span::styled(
                        active_marker,
                        Style::default().fg(self.theme.success).bold(),
                    ),
                ]);

                let style = if is_selected {
                    Style::default().bg(self.theme.surface)
                } else {
                    Style::default()
                };

                ListItem::new(content).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render detail view for selected run
    fn render_detail(&self, frame: &mut Frame, area: Rect) {
        if self.runs.is_empty() {
            return;
        }

        let selected = self.selected.min(self.runs.len().saturating_sub(1));
        let run = &self.runs[selected];
        let title = format!("Run Details: {}", run.run_id);
        let block = Panel::new(&self.theme).title(&title).block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let mut lines = vec![
            Line::from(vec![
                Span::styled("Experiment: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(&run.experiment, Style::default().fg(self.theme.text_primary)),
            ]),
            Line::from(vec![
                Span::styled("Date: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(&run.date, Style::default().fg(self.theme.text_primary)),
            ]),
            Line::from(vec![
                Span::styled("Git SHA: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(&run.git_sha, Style::default().fg(self.theme.text_secondary)),
            ]),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(
                    &run.status,
                    Style::default().fg(match run.status.as_str() {
                        "completed" => self.theme.success,
                        "failed" => self.theme.error,
                        _ => self.theme.text_muted,
                    }),
                ),
            ]),
            Line::from(vec![
                Span::styled("Best AUC: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(
                    run.best_auc.map_or("-".to_string(), |a| format!("{:.4}", a)),
                    Style::default().fg(self.theme.accent),
                ),
            ]),
            Line::from(vec![
                Span::styled("Avg AUC: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(
                    run.avg_auc.map_or("-".to_string(), |a| format!("{:.4}", a)),
                    Style::default().fg(self.theme.text_secondary),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Targets:",
                Style::default().fg(self.theme.text_muted).bold(),
            )),
        ];

        // Add targets with AUC
        let visible_height = inner.height as usize - lines.len();
        let target_start = self.detail_scroll;
        let target_end = (target_start + visible_height).min(run.targets.len());

        for target in &run.targets[target_start..target_end] {
            let auc_text = target.auc.map_or("-".to_string(), |a| format!("{:.4}", a));
            let status_color = match target.status.as_str() {
                "completed" | "trained" => self.theme.success,
                "failed" => self.theme.error,
                _ => self.theme.text_muted,
            };

            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("{:30}", target.name),
                    Style::default().fg(self.theme.text_primary),
                ),
                Span::styled(
                    format!(" {:>7}", auc_text),
                    Style::default().fg(self.theme.accent),
                ),
                Span::styled(
                    format!(" {:12}", target.best_family),
                    Style::default().fg(self.theme.text_secondary),
                ),
                Span::styled(
                    format!(" {:9}", target.status),
                    Style::default().fg(status_color),
                ),
            ]));
        }

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, inner);
    }

    /// Render footer
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = if self.show_detail {
            vec![
                ("[j/k]", "Scroll"),
                ("[Enter]", "Back"),
                ("[a]", "Set Active"),
                ("[q/Esc]", "Back"),
            ]
        } else {
            vec![
                ("[j/k]", "Select"),
                ("[Enter]", "Details"),
                ("[a]", "Set Active"),
                ("[r]", "Refresh"),
                ("[q/Esc]", "Back"),
            ]
        };

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  ", Style::default()));
            }
            spans.push(Span::styled(*key, Style::default().fg(self.theme.accent)));
            spans.push(Span::styled(
                format!(" {}", desc),
                Style::default().fg(self.theme.text_muted),
            ));
        }

        // Show message or run count
        spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));

        if let Some((msg, is_error)) = &self.message {
            spans.push(Span::styled(
                msg,
                Style::default().fg(if *is_error {
                    self.theme.error
                } else {
                    self.theme.success
                }),
            ));
        } else {
            spans.push(Span::styled(
                format!("{} runs", self.runs.len()),
                Style::default().fg(self.theme.text_muted),
            ));
        }

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for ModelSelectorView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for ModelSelectorView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50), // Run list
                Constraint::Percentage(50), // Detail view
                Constraint::Length(1),      // Footer
            ])
            .margin(1)
            .split(area);

        // Recalculate to get footer at bottom
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),    // Content
                Constraint::Length(1), // Footer
            ])
            .margin(1)
            .split(area);

        let content_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(50), // List
                Constraint::Percentage(50), // Detail
            ])
            .split(main_chunks[0]);

        self.render_list(frame, content_chunks[0]);
        self.render_detail(frame, content_chunks[1]);
        self.render_footer(frame, main_chunks[1]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<bool> {
        // Clear message on key
        self.message = None;

        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.show_detail {
                    self.show_detail = false;
                } else {
                    return Ok(true); // Go back
                }
            }
            // Navigation
            KeyCode::Up | KeyCode::Char('k') => {
                if self.show_detail {
                    self.detail_scroll = self.detail_scroll.saturating_sub(1);
                } else if self.selected > 0 {
                    self.selected -= 1;
                    // Update scroll to keep selection visible
                    if self.selected < self.scroll {
                        self.scroll = self.selected;
                    }
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.show_detail {
                    if !self.runs.is_empty() {
                        let idx = self.selected.min(self.runs.len().saturating_sub(1));
                        let max_scroll = self.runs[idx].targets.len().saturating_sub(5);
                        if self.detail_scroll < max_scroll {
                            self.detail_scroll += 1;
                        }
                    }
                } else if self.selected < self.runs.len().saturating_sub(1) {
                    self.selected += 1;
                }
            }
            KeyCode::Enter => {
                if self.show_detail {
                    self.show_detail = false;
                } else {
                    self.show_detail = true;
                    self.detail_scroll = 0;
                }
            }
            KeyCode::Char('a') => {
                // Set as active
                match self.set_active_run() {
                    Ok(msg) => self.message = Some((msg, false)),
                    Err(e) => self.message = Some((format!("Error: {}", e), true)),
                }
            }
            KeyCode::Char('r') => {
                // Refresh
                self.scan_runs();
                self.load_active_run();
                self.message = Some(("Refreshed run list".to_string(), false));
            }
            _ => {}
        }

        Ok(false)
    }
}
