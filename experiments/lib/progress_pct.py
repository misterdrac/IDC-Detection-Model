"""Terminal progress as integer percentages only (no bars)."""
from __future__ import annotations


def report_pct(label: str, current: int, total: int, state: dict) -> None:
    """Print `label: N%` when the integer percent changes."""
    if total <= 0:
        return
    pct = min(100, (100 * current) // total)
    if state.get(label) != pct:
        state[label] = pct
        print(f"{label}: {pct}%", flush=True)
