# Dashboard Fixes Round 3: Cleanup

**Branch**: `analysis/code-review-and-raw-ohlcv`
**Status**: Complete
**Scope**: 3 files modified (server.py, alpaca_stream.py, model_selector.rs)

---

## Issues (7 total: 2 MEDIUM, 5 LOW)

### MEDIUM

**1. Decision statistics mismatch** — `server.py:630-651`
- Compute `trade_count`/`hold_count`/`blocked_count` BEFORE the `[:limit]` so they reflect the full filtered set, matching `total_decisions`.

**2. Type coercion crash in decision explanation** — `server.py:720-750`
- Wrap numeric format strings with `float()` coercion or try/except to handle non-numeric trace values gracefully.

### LOW

**3. Model selector hardcoded paths** — `model_selector.rs:250,260`
- Replace `PathBuf::from("LIVE_TRADING/...")` with `crate::config::project_root().join("LIVE_TRADING/...")`.

**4. `get_recent_events(count=0)` returns all** — `alpaca_stream.py:177`
- Guard: `if count <= 0: return []`

**5. Timestamp sorting empty string default** — `server.py:517,627,687`
- Use `"0"` as default instead of `""` so missing timestamps sort to the end (oldest) in reverse sort.

**6. Silent date parse error** — `server.py:509-514`
- Log a warning when `since` parameter fails to parse.

**7. Incomplete shutdown handler** — `server.py:241-243`
- Remove misleading comment; add a log message noting shutdown.

---

## Verification
- `cargo build --release` passes
- Bridge starts without errors
