# Phase 3: Rust TUI Defensive Coding

**Status**: Pending
**Parent**: `dashboard-bug-fixes-master.md`
**Scope**: 4 files, ~30 lines changed
**Depends on**: Nothing

## Problem

Several Rust modules have edge cases that can panic: unchecked array indexing, `unwrap()` on fallible operations, unsafe UTF-8 byte slicing, and unsigned integer underflow on small terminals.

## Changes

### 3a. Bounds-safe indexing (`model_selector.rs`, bug 9 — HIGH)

**Lines 281, 395**: `self.runs[self.selected]` can panic if `selected` is stale after a rescan clears `runs`.

Add a bounds clamp at the start of `set_active_run()`, `render_detail()`, and any other method that indexes `self.runs`:

```rust
// After the is_empty() check, add:
if self.selected >= self.runs.len() {
    self.selected = self.runs.len().saturating_sub(1);
}
let run = &self.runs[self.selected];
```

Also add this clamp in `scan_runs()` after rebuilding the runs list:

```rust
fn scan_runs(&mut self) {
    // ... rebuild self.runs ...
    if self.selected >= self.runs.len() {
        self.selected = self.runs.len().saturating_sub(1);
    }
}
```

### 3b. Fallible client construction (`client.rs`, bug 10 — MEDIUM)

**Line 29**: Replace `unwrap()` with `?`.

```rust
pub fn new(base_url: &str) -> Result<Self> {
    Ok(Self {
        http_url: format!("http://{}", base_url),
        ws_url: format!("ws://{}", base_url),
        client: Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?,
    })
}
```

Update all call sites: `DashboardClient::new(url)` → `DashboardClient::new(url)?`

Search for call sites with: `grep -rn "DashboardClient::new" DASHBOARD/dashboard/src/`

### 3c. Log position deserialization failures (`client.rs`, bug 11 — MEDIUM)

**Line 59**: Add warning log instead of silent drop.

```rust
.filter_map(|p| {
    serde_json::from_value(p.clone())
        .map_err(|e| tracing::warn!("Failed to parse position: {}", e))
        .ok()
})
```

### 3d. Safe timestamp display (`events.rs`, bug 12 — MEDIUM)

**Line 91**: Replace byte indexing with `.get()`.

```rust
pub fn display_timestamp(&self) -> String {
    if self.timestamp.len() >= 19 {
        self.timestamp.get(11..19)
            .unwrap_or(&self.timestamp)
            .to_string()
    } else {
        self.timestamp.clone()
    }
}
```

### 3e. Safe notification positioning (`notification.rs`, bug 13 — MEDIUM)

**Line 268**: Use saturating arithmetic.

```rust
let notification_area = Rect {
    x: area.x.saturating_add(area.width.saturating_sub(notification_width + 2)),
    y: area.y + y_offset,
    width: notification_width,
    height: notification_height,
};
```

## Verification

- [ ] `cargo build --release` succeeds
- [ ] No `unwrap()` on `Client::builder().build()`
- [ ] `self.runs[self.selected]` always preceded by bounds clamp
- [ ] Malformed position JSON logged at warn level
- [ ] Timestamp slicing uses `.get()` not `[]`
- [ ] Notification area uses `saturating_sub`/`saturating_add`

## Files Changed

1. `DASHBOARD/dashboard/src/views/model_selector.rs` — bounds clamping
2. `DASHBOARD/dashboard/src/api/client.rs` — fallible new(), log deser failures
3. `DASHBOARD/dashboard/src/api/events.rs` — safe string slicing
4. `DASHBOARD/dashboard/src/ui/notification.rs` — safe coordinate arithmetic
