"""runs/leaderboard.json: best finished runs per course and controller mode.

Format::

    {"version": 1,
     "courses": {"gauntlet": {"hybrid": [{"total_time_s": 21.3, ...}, ...],
                              "cpg": [...]}}}

Entries are sorted by ``total_time_s`` (race time + penalties); only finished
runs are ranked, at most ``keep`` per (course, mode).
"""

from __future__ import annotations

import json
from pathlib import Path

CONTROLLER_MODES = ("hybrid", "cpg", "brain-steer", "policy")


def controller_mode(cfg) -> str:
    """Leaderboard key for an ``AppConfig``: brain-steer, hybrid or cpg."""
    brain = getattr(cfg, "brain", None)
    if brain is not None and getattr(brain, "enabled", False) and getattr(brain, "steer", False):
        return "brain-steer"
    return cfg.controller.kind


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"version": 1, "courses": {}}
    data = json.loads(p.read_text())
    data.setdefault("courses", {})
    return data


def add_entry(path: str | Path, course: str, mode: str, entry: dict,
              keep: int = 20) -> int | None:
    """Insert a finished run; returns its 1-based rank (None if not kept)."""
    p = Path(path)
    data = load(p)
    board = data["courses"].setdefault(course, {}).setdefault(mode, [])
    board.append(entry)
    board.sort(key=lambda e: e["total_time_s"])
    rank = next((i + 1 for i, e in enumerate(board) if e is entry), None)
    del board[keep:]
    if rank is not None and rank > keep:
        rank = None
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)
    return rank


def best(path: str | Path, course: str, mode: str) -> dict | None:
    board = load(path)["courses"].get(course, {}).get(mode, [])
    return board[0] if board else None


def format_board(path: str | Path, course: str | None = None, n: int = 5) -> str:
    data = load(path)["courses"]
    lines = []
    for c in sorted(data) if course is None else [course]:
        for mode, board in sorted(data.get(c, {}).items()):
            lines.append(f"{c} / {mode}:")
            for i, e in enumerate(board[:n]):
                lines.append(f"  {i + 1:2d}. {e['total_time_s']:7.2f}s  (race {e['time_s']:.2f}s"
                             f" + pen {e['penalty_s']:.1f}s, falls {e['n_falls']}, "
                             f"respawns {e['n_respawns']})  {e.get('date', '')}")
    return "\n".join(lines) if lines else "(leaderboard empty)"
