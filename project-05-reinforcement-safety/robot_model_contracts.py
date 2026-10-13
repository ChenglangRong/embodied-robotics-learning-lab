"""Offline contract exercises. These functions perform zero model/API calls."""
import numpy as np


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("skill") not in {"reach", "hold", "stop"}:
        raise ValueError("whitelisted skill required")
    if set(plan)-{"skill", "goal_xy", "timeout_steps", "reason"}:
        raise ValueError("unknown planning fields")
    timeout = plan.get("timeout_steps", 150)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 150:
        raise ValueError("timeout_steps must be 1..150")
    if plan["skill"] == "reach":
        goal = np.asarray(plan.get("goal_xy", []), dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all() or not .2 <= np.linalg.norm(goal) <= 1.:
            raise ValueError("finite geometrically reachable planar goal required")
    return {**plan, "timeout_steps": timeout}


def validate_chunk(packet, action_dim, now_s, period_s=.02, max_age_s=.5):
    """Shape/unit/time/bounds checks; mechanical safety remains an execution check."""
    chunk = np.asarray(packet["actions"], dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != action_dim or not 1 <= chunk.shape[0] <= 50:
        raise ValueError("action chunk shape must be [1..50, action_dim]")
    if not np.isfinite(chunk).all() or np.any(np.abs(chunk)>1):
        raise ValueError("finite normalized actions in [-1,1] required")
    if packet.get("units") != "normalized_velocity" or packet.get("period_s") != period_s:
        raise ValueError("declared units and control period must match the adapter")
    timestamp = packet.get("timestamp_s")
    if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or not np.isfinite(now_s) or not 0 <= now_s-timestamp <= max_age_s:
        raise ValueError("stale or future observation/action packet")
    replan = packet.get("execute_before_replan", 1)
    if isinstance(replan, bool) or not isinstance(replan, int) or not 1 <= replan <= len(chunk):
        raise ValueError("valid executed prefix length required")
    return chunk[:replan].copy()
