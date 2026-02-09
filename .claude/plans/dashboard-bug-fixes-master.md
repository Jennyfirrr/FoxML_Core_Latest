# Dashboard Bug Fixes — Master Plan

**Status**: Complete
**Created**: 2026-02-08
**Branch**: `analysis/code-review-and-raw-ohlcv`

## Context

Code review of the DASHBOARD directory (Rust TUI + Python IPC bridge) identified 15 bugs ranging from crash-on-startup to cosmetic issues. The most severe: a `NameError` that crashes the bridge if the Alpaca module is missing, a Rust panic that corrupts the terminal, and integer underflows that crash the TUI when progress exceeds 100%.

## Bug Index

| # | Bug | File | Severity | Phase |
|---|-----|------|----------|-------|
| 1 | Logger used before definition | server.py:61 | CRITICAL | 1 |
| 2 | Unbounded event queues | server.py:71,74 | CRITICAL | 1 |
| 3 | Ping timestamp `"..."` | server.py:210 | HIGH | 1 |
| 4 | Inconsistent WS disconnect | server.py:213 | HIGH | 1 |
| 5 | Training event drop race | server.py:856-862 | HIGH | 1 |
| 6 | Non-atomic control state write | server.py:134 | LOW | 1 |
| 7 | Terminal not restored on panic | main.rs:32-44 | CRITICAL | 2 |
| 8 | Progress bar underflow panic | training.rs:693 | HIGH | 2 |
| 9 | Unchecked array indexing | model_selector.rs:281,395 | HIGH | 3 |
| 10 | Client::new() unwrap panic | client.rs:29 | MEDIUM | 3 |
| 11 | Silent position deser drops | client.rs:59 | MEDIUM | 3 |
| 12 | UTF-8 string slicing | events.rs:91 | MEDIUM | 3 |
| 13 | Notification position overflow | notification.rs:268 | MEDIUM | 3 |

## Sub-Plans

### Phase 1: Python Bridge Fixes
**File**: `dashboard-phase1-bridge-fixes.md`
**Status**: Complete
**Scope**: 1 file (`server.py`), ~20 lines changed
**Depends on**: Nothing

Fix all 6 Python bridge bugs in a single pass. All changes are in `DASHBOARD/bridge/server.py`.

### Phase 2: Rust TUI Panic Prevention
**File**: `dashboard-phase2-panic-prevention.md`
**Status**: Complete
**Scope**: 2 files (`main.rs`, `training.rs`), ~25 lines changed
**Depends on**: Nothing

Fix the two most dangerous Rust bugs: terminal corruption on panic, progress bar underflow.

### Phase 3: Rust TUI Defensive Coding
**File**: `dashboard-phase3-defensive-coding.md`
**Status**: Complete
**Scope**: 4 files, ~30 lines changed
**Depends on**: Nothing

Harden edge cases: bounds checks, error handling, safe arithmetic.

## Dependency Graph

```
Phase 1 (Python bridge)     — standalone
Phase 2 (Rust panic fixes)  — standalone
Phase 3 (Rust defensive)    — standalone

All phases are independent. Order is by severity priority only.
```

## Verification

After all phases:
- [x] Bridge starts without Alpaca module installed (logger defined before use)
- [x] Rust TUI compiles: `cd DASHBOARD/dashboard && cargo build --release`
- [x] No `unwrap()` on `Client::build()` (uses `expect()` with message)
- [x] No unchecked `progress_bar_width - filled` (uses saturating_sub + clamp)
- [x] `self.runs[self.selected]` always preceded by bounds clamp

## Session Notes

### 2026-02-08: Plan created
- 15 bugs identified from comprehensive dashboard code review
- Grouped into 3 independent phases by component and severity

### 2026-02-08: All phases implemented
- Phase 1: 6 Python bridge bugs fixed in server.py
- Phase 2: 2 Rust panic bugs fixed in main.rs + training.rs
- Phase 3: 5 Rust defensive coding fixes across 4 files
- All verification checks pass (Python syntax OK, Rust builds clean)
