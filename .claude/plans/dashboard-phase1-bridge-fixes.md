# Phase 1: Python Bridge Fixes

**Status**: Pending
**Parent**: `dashboard-bug-fixes-master.md`
**Scope**: 1 file, ~20 lines changed
**Depends on**: Nothing

## Problem

The Python IPC bridge (`DASHBOARD/bridge/server.py`) has 6 bugs including a crash-on-startup `NameError`, unbounded queues that leak memory, a placeholder timestamp, inconsistent error handling, a race condition, and a non-atomic file write.

## Changes

All changes in `DASHBOARD/bridge/server.py`.

### 1a. Move logger definition above imports (bug 1 — CRITICAL)

**Lines 31 → 63**: `logger` is used at line 61 but defined at line 63. Move definition to after `logging.basicConfig()` (after line 31).

```python
# After line 31 (after logging.basicConfig block), ADD:
logger = logging.getLogger(__name__)

# DELETE the duplicate at line 63:
# logger = logging.getLogger(__name__)
```

### 1b. Bound event queues (bug 2 — CRITICAL)

**Lines 71, 74**: Add `maxsize=1000` to match `training_event_queue` at line 849.

```python
# Line 71:
event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
# Line 74:
alpaca_event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
```

### 1c. Fix ping timestamp (bug 3 — HIGH)

**Line 210**: Replace `"..."` literal with actual ISO timestamp.

```python
await websocket.send_json({"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()})
```

### 1d. Safe WebSocket disconnect (bug 4 — HIGH)

**Line 213**: Add membership check before `remove()`.

```python
except WebSocketDisconnect:
    if websocket in active_connections:
        active_connections.remove(websocket)
    logger.info(f"WebSocket client disconnected. Total connections: {len(active_connections)}")
```

### 1e. Fix training event overflow (bug 5 — HIGH)

**Lines 856-862**: Replace get-then-put race with simple drop-newest.

```python
def on_training_event(event: Dict[str, Any]) -> None:
    """Callback for training progress events."""
    try:
        training_event_queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.debug("Training event queue full, dropping event")
```

### 1f. Atomic control state write (bug 6 — LOW)

**Lines 127-138**: Write to temp file then rename.

```python
def _write_control_state(state: Dict[str, Any]) -> None:
    try:
        CONTROL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state["last_updated"] = datetime.now().isoformat()
        tmp = CONTROL_STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.rename(CONTROL_STATE_FILE)
    except Exception as e:
        logger.error(f"Error writing control state: {e}")
        raise
```

## Verification

- [ ] `python -c "from DASHBOARD.bridge import server"` succeeds without alpaca_stream
- [ ] `event_queue.maxsize == 1000`
- [ ] Ping messages contain valid ISO timestamps
- [ ] No `ValueError` on concurrent WebSocket disconnect
- [ ] Training event queue overflow logs debug message, no race

## Files Changed

1. `DASHBOARD/bridge/server.py` — all 6 fixes
