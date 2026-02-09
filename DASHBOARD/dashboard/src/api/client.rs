//! HTTP/WebSocket client for IPC bridge

use anyhow::Result;
use futures::{SinkExt, StreamExt};
use reqwest::Client;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::alpaca::{AlpacaEvent, AlpacaStatus, BridgeHealth};
use super::events::{Position, RiskStatus, TradingEvent, TrainingEvent};

/// Dashboard API client
pub struct DashboardClient {
    http_url: String,
    ws_url: String,
    client: Client,
}

impl DashboardClient {
    /// Create new client
    pub fn new(base_url: &str) -> Self {
        Self {
            http_url: format!("http://{}", base_url),
            ws_url: format!("ws://{}", base_url),
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .expect("Failed to build HTTP client (TLS backend unavailable?)"),
        }
    }

    /// Get metrics
    pub async fn get_metrics(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/metrics", self.http_url);
        let response = self.client.get(&url).send().await?;
        let metrics = response.json().await?;
        Ok(metrics)
    }

    /// Get state
    pub async fn get_state(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/state", self.http_url);
        let response = self.client.get(&url).send().await?;
        let state = response.json().await?;
        Ok(state)
    }

    /// Get positions
    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        let url = format!("{}/api/positions", self.http_url);
        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let positions = json["positions"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| {
                        serde_json::from_value(p.clone())
                            .map_err(|e| tracing::warn!("Failed to parse position: {}", e))
                            .ok()
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(positions)
    }

    /// Get risk status
    pub async fn get_risk_status(&self) -> Result<RiskStatus> {
        let url = format!("{}/api/risk/status", self.http_url);
        let response = self.client.get(&url).send().await?;
        let status = response.json().await?;
        Ok(status)
    }

    /// Get recent decisions
    pub async fn get_recent_decisions(&self, limit: usize) -> Result<serde_json::Value> {
        let url = format!("{}/api/decisions/recent?limit={}", self.http_url, limit);
        let response = self.client.get(&url).send().await?;
        let decisions = response.json().await?;
        Ok(decisions)
    }

    /// Get WebSocket URL for events
    pub fn ws_events_url(&self) -> String {
        format!("{}/ws/events", self.ws_url)
    }

    /// Connect to events WebSocket and return a receiver channel
    pub async fn connect_events_ws(&self) -> Result<mpsc::Receiver<TradingEvent>> {
        let url = self.ws_events_url();
        let (ws_stream, _) = connect_async(&url).await?;
        let (mut write, mut read) = ws_stream.split();

        let (tx, rx) = mpsc::channel(100);

        // Spawn task to read from WebSocket
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        // Skip ping messages
                        if text.contains("\"type\":\"ping\"") {
                            continue;
                        }

                        match serde_json::from_str::<TradingEvent>(&text) {
                            Ok(event) => {
                                if tx.send(event).await.is_err() {
                                    break; // Receiver dropped
                                }
                            }
                            Err(e) => {
                                // Log parse error but continue
                                tracing::debug!("Failed to parse event: {} - {}", e, text);
                            }
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        // Respond to ping
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Ok(Message::Close(_)) => {
                        break;
                    }
                    Err(e) => {
                        tracing::error!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(rx)
    }

    /// Pause trading engine
    pub async fn pause_engine(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/control/pause", self.http_url);
        let response = self.client.post(&url).send().await?;
        let result = response.json().await?;
        Ok(result)
    }

    /// Resume trading engine
    pub async fn resume_engine(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/control/resume", self.http_url);
        let response = self.client.post(&url).send().await?;
        let result = response.json().await?;
        Ok(result)
    }

    /// Toggle kill switch
    pub async fn toggle_kill_switch(&self, enable: bool, reason: Option<String>) -> Result<serde_json::Value> {
        let url = format!("{}/api/control/kill_switch", self.http_url);
        let body = serde_json::json!({
            "action": if enable { "enable" } else { "disable" },
            "reason": reason,
        });
        let response = self.client
            .post(&url)
            .json(&body)
            .send()
            .await?;
        let result = response.json().await?;
        Ok(result)
    }

    /// Get control status
    pub async fn get_control_status(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/control/status", self.http_url);
        let response = self.client.get(&url).send().await?;
        let status = response.json().await?;
        Ok(status)
    }

    // Alpaca endpoints

    /// Get Alpaca connection status
    pub async fn get_alpaca_status(&self) -> Result<AlpacaStatus> {
        let url = format!("{}/api/alpaca/status", self.http_url);
        let response = self.client.get(&url).send().await?;
        let status = response.json().await?;
        Ok(status)
    }

    /// Get recent Alpaca events
    pub async fn get_alpaca_events(&self, count: usize) -> Result<Vec<AlpacaEvent>> {
        let url = format!("{}/api/alpaca/events/recent?count={}", self.http_url, count);
        let response = self.client.get(&url).send().await?;
        let events = response.json().await?;
        Ok(events)
    }

    /// Get WebSocket URL for Alpaca events
    pub fn ws_alpaca_url(&self) -> String {
        format!("{}/ws/alpaca", self.ws_url)
    }

    /// Get WebSocket URL for training events
    pub fn ws_training_url(&self) -> String {
        format!("{}/ws/training", self.ws_url)
    }

    /// Connect to training WebSocket and return a receiver channel
    pub async fn connect_training_ws(&self) -> Result<mpsc::Receiver<TrainingEvent>> {
        let url = self.ws_training_url();
        let (ws_stream, _) = connect_async(&url).await?;
        let (mut write, mut read) = ws_stream.split();

        let (tx, rx) = mpsc::channel(100);

        // Spawn task to read from WebSocket
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        // Skip ping messages
                        if text.contains("\"type\":\"ping\"") {
                            continue;
                        }

                        match serde_json::from_str::<TrainingEvent>(&text) {
                            Ok(event) => {
                                if tx.send(event).await.is_err() {
                                    break; // Receiver dropped
                                }
                            }
                            Err(e) => {
                                tracing::debug!("Failed to parse training event: {} - {}", e, text);
                            }
                        }
                    }
                    Ok(Message::Ping(data)) => {
                        let _ = write.send(Message::Pong(data)).await;
                    }
                    Ok(Message::Close(_)) => {
                        break;
                    }
                    Err(e) => {
                        tracing::error!("Training WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(rx)
    }

    /// Get bridge health status
    pub async fn get_health(&self) -> Result<BridgeHealth> {
        let url = format!("{}/health", self.http_url);
        let response = self.client.get(&url).send().await?;
        let health = response.json().await?;
        Ok(health)
    }
}
