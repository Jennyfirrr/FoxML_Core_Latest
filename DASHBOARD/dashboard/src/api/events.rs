//! Event handling from WebSocket stream

use serde::{Deserialize, Serialize};

/// Event from trading engine via WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingEvent {
    pub event_type: String,
    pub timestamp: String,
    #[serde(default)]
    pub severity: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

impl TradingEvent {
    /// Get display message for the event
    pub fn display_message(&self) -> String {
        if !self.message.is_empty() {
            self.message.clone()
        } else if let Some(msg) = self.data.get("message").and_then(|v| v.as_str()) {
            msg.to_string()
        } else {
            // Format based on event type
            match self.event_type.as_str() {
                "TRADE_FILLED" => {
                    let symbol = self.data.get("symbol").and_then(|v| v.as_str()).unwrap_or("???");
                    let side = self.data.get("side").and_then(|v| v.as_str()).unwrap_or("???");
                    let shares = self.data.get("shares").and_then(|v| v.as_i64()).unwrap_or(0);
                    format!("{} {} {} shares", side, symbol, shares)
                }
                "DECISION_MADE" => {
                    let symbol = self.data.get("symbol").and_then(|v| v.as_str()).unwrap_or("???");
                    let decision = self.data.get("decision").and_then(|v| v.as_str()).unwrap_or("???");
                    format!("{}: {}", symbol, decision)
                }
                "CYCLE_START" => {
                    let cycle = self.data.get("cycle_id").and_then(|v| v.as_i64()).unwrap_or(0);
                    format!("Cycle {} started", cycle)
                }
                "CYCLE_END" => {
                    let cycle = self.data.get("cycle_id").and_then(|v| v.as_i64()).unwrap_or(0);
                    let duration = self.data.get("duration_seconds").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    format!("Cycle {} completed ({:.2}s)", cycle, duration)
                }
                "STAGE_CHANGE" => {
                    let stage = self.data.get("stage").and_then(|v| v.as_str()).unwrap_or("???");
                    let symbol = self.data.get("symbol").and_then(|v| v.as_str());
                    if let Some(sym) = symbol {
                        format!("Stage: {} ({})", stage, sym)
                    } else {
                        format!("Stage: {}", stage)
                    }
                }
                "KILL_SWITCH_TRIGGERED" => {
                    let reason = self.data.get("reason").and_then(|v| v.as_str()).unwrap_or("unknown");
                    format!("KILL SWITCH: {}", reason)
                }
                "ERROR" => {
                    let error = self.data.get("error").and_then(|v| v.as_str()).unwrap_or("unknown error");
                    format!("Error: {}", error)
                }
                _ => {
                    format!("{}", self.event_type)
                }
            }
        }
    }

    /// Get severity level (for coloring)
    pub fn severity_level(&self) -> &str {
        if !self.severity.is_empty() {
            &self.severity
        } else {
            // Infer from event type
            match self.event_type.as_str() {
                "ERROR" | "KILL_SWITCH_TRIGGERED" | "TRADE_REJECTED" => "error",
                "WARNING" | "RISK_WARNING" => "warning",
                "TRADE_FILLED" | "DECISION_MADE" => "info",
                _ => "debug",
            }
        }
    }

    /// Extract timestamp for display (HH:MM:SS)
    pub fn display_timestamp(&self) -> String {
        if self.timestamp.len() >= 19 {
            // ISO format: 2025-01-21T10:30:00...
            self.timestamp.get(11..19)
                .unwrap_or(&self.timestamp)
                .to_string()
        } else {
            self.timestamp.clone()
        }
    }
}

/// Position data from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub shares: i64,
    pub entry_price: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_pct: f64,
    pub weight: f64,
    #[serde(default)]
    pub entry_time: Option<String>,
    #[serde(default)]
    pub side: String,
}

/// Risk status from API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiskStatus {
    pub trading_allowed: bool,
    pub kill_switch_active: bool,
    #[serde(default)]
    pub kill_switch_reason: Option<String>,
    #[serde(default)]
    pub daily_pnl_pct: f64,
    #[serde(default)]
    pub daily_loss_limit_pct: f64,
    #[serde(default)]
    pub daily_loss_remaining_pct: f64,
    #[serde(default)]
    pub drawdown_pct: f64,
    #[serde(default)]
    pub max_drawdown_limit_pct: f64,
    #[serde(default)]
    pub drawdown_remaining_pct: f64,
    #[serde(default)]
    pub gross_exposure: f64,
    #[serde(default)]
    pub net_exposure: f64,
    #[serde(default)]
    pub max_gross_exposure: f64,
    #[serde(default)]
    pub warnings: Vec<RiskWarning>,
}

/// Risk warning from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskWarning {
    #[serde(rename = "type")]
    pub warning_type: String,
    pub message: String,
    pub severity: String,
    pub threshold_pct: f64,
    pub current_pct: f64,
}

/// Training progress event from WebSocket/file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEvent {
    pub event_type: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub stage: String,
    // stage_change events use "new_stage" instead of "stage"
    #[serde(default)]
    pub new_stage: Option<String>,
    #[serde(default)]
    pub previous_stage: Option<String>,
    #[serde(default)]
    pub progress_pct: f64,
    #[serde(default)]
    pub current_target: Option<String>,
    #[serde(default)]
    pub targets_complete: i64,
    #[serde(default)]
    pub targets_total: i64,
    #[serde(default)]
    pub message: Option<String>,
    // For target_start/target_complete events
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub target_index: Option<i64>,
    #[serde(default)]
    pub total_targets: Option<i64>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub models_trained: Option<i64>,
    #[serde(default)]
    pub best_auc: Option<f64>,
    #[serde(default)]
    pub duration_seconds: Option<f64>,
    // For run_complete events
    #[serde(default)]
    pub successful_targets: Option<i64>,
    // For error events
    #[serde(default)]
    pub error_message: Option<String>,
    #[serde(default)]
    pub error_type: Option<String>,
    #[serde(default)]
    pub recoverable: Option<bool>,
}

impl TrainingEvent {
    /// Get effective stage (handles stage_change events which use new_stage)
    pub fn effective_stage(&self) -> &str {
        // For stage_change events, use new_stage; otherwise use stage
        if self.event_type == "stage_change" {
            self.new_stage.as_deref().unwrap_or(&self.stage)
        } else {
            &self.stage
        }
    }

    /// Get display message for the event
    pub fn display_message(&self) -> String {
        if let Some(msg) = &self.message {
            return msg.clone();
        }
        match self.event_type.as_str() {
            "progress" => {
                if let Some(target) = &self.current_target {
                    format!("[{:.0}%] {} - {}", self.progress_pct, self.effective_stage(), target)
                } else {
                    format!("[{:.0}%] {}", self.progress_pct, self.effective_stage())
                }
            }
            "stage_change" => format!("Stage: {}", self.effective_stage()),
            "target_start" => {
                format!("Started: {}", self.target.as_deref().unwrap_or("???"))
            }
            "target_complete" => {
                let status = self.status.as_deref().unwrap_or("done");
                let target = self.target.as_deref().unwrap_or("???");
                if let Some(auc) = self.best_auc {
                    format!("{}: {} (AUC: {:.4})", target, status, auc)
                } else {
                    format!("{}: {}", target, status)
                }
            }
            "run_complete" => {
                let status = self.status.as_deref().unwrap_or("complete");
                format!("Run {}: {}", status, self.run_id)
            }
            "error" => {
                self.error_message.clone().unwrap_or_else(|| "Error".to_string())
            }
            _ => self.event_type.clone(),
        }
    }
}
