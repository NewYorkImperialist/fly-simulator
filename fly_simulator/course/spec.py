"""Course file format (JSON, or TOML with the same structure).

A course is an ordered list of sections laid out along +x from the spawn point
(x = 0). Example (see docs/COURSE.md for every section type and parameter)::

    {
      "name": "tutorial",
      "description": "gentle intro",
      "sections": [
        {"type": "flat", "length": 4},
        {"type": "bumps", "length": 12, "params": {"count": 4}},
        {"type": "checkpoint"},
        {"type": "stairs", "params": {"steps_up": 2, "step_height": 0.08}},
        {"type": "finish"}
      ]
    }

Top-level race / layout options (all optional) are the fields of ``CourseSpec``.
Section ``length`` is a minimum: sections whose geometry needs more room (stairs,
gaps, slalom, ...) grow to fit. A ``finish`` section is appended if missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

COURSES_DIR = Path(__file__).resolve().parents[2] / "courses"

# Every section type with its default params. Unknown params are an error (typos).
SECTION_DEFAULTS: dict[str, dict] = {
    "flat": {},
    "bumps": {"count": 4, "height": 0.15, "radius": 1.6, "spread": 2.0},
    "rubble": {"rocks": 6, "lumps": 14, "rock_height": 0.15, "rock_radius": 0.35,
               "lump_height": 0.05, "lump_radius": 0.5, "spread": 2.5},
    "blocks": {"count": 5, "height": 0.12, "size": 1.2, "spread": 2.5},
    "stairs": {"steps_up": 3, "steps_down": 3, "step_height": 0.1, "step_depth": 1.6,
               "landing": 2.0},
    "gap": {"depth": 0.25, "width": 0.25, "plateau": 2.0, "ramp_angle_deg": 8.0},
    "dip": {"depth": 0.15, "width": 2.0, "plateau": 2.0, "ramp_angle_deg": 8.0},
    "ramp": {"height": 0.3, "angle_deg": 7.0, "top": 2.0},
    "pillars": {"count": 5, "spacing": 9.0, "offset": 2.6, "size": 0.6, "height": 3.0,
                "lead": 6.0, "first_side": "left"},
    "tunnel": {"ceiling_z": 1.9, "thickness": 0.15, "wall_offset": 3.5, "piece": 4.0},
    "whip_gauntlet": {"cracks": 3, "level": 2, "sides": ["left", "right", "random"]},
    "loom_zone": {"set": "LC4", "duration_s": 0.5, "repeats": 1, "interval_s": 1.0},
    "checkpoint": {"pad": 3.0},
    "finish": {"pad": 3.0},
}
# Sections whose length is derived from their params when "length" is omitted.
DEFAULT_LENGTH = {"flat": 4.0, "bumps": 12.0, "rubble": 12.0, "blocks": 12.0,
                  "tunnel": 12.0, "whip_gauntlet": 18.0, "loom_zone": 10.0}


@dataclass
class SectionSpec:
    type: str
    length: float = 0.0  # minimum length (mm); 0 = whatever the geometry needs
    params: dict = field(default_factory=dict)
    name: str = ""

    def param(self, key: str):
        return self.params.get(key, SECTION_DEFAULTS[self.type][key])


@dataclass
class CourseSpec:
    name: str
    sections: list[SectionSpec]
    description: str = ""
    seed: int = 0  # random placement inside rubble / bumps / blocks sections
    track_half_width: float = 8.0  # mm; full-width features span |y| <= this
    # Chunk length used while the course is installed (pool per chunk is fixed, so
    # shorter chunks = denser geometry; the load window covers 1 behind, 5 ahead).
    chunk_length: float = 6.0
    start_x: float = 2.0  # start line (timer starts when the thorax crosses it)
    lead_in: float = 4.0  # flat ground from the spawn point to the first section
    timeout_s: float = 120.0  # race time after which the run is DNF
    fall_penalty_s: float = 3.0  # added to the time per fall
    respawn_after_s: float = 3.0  # FALLEN this long -> respawn at the last checkpoint
    respawn_penalty_s: float = 0.0  # extra per respawn (the fall is already penalised)
    max_respawns: int = 10  # more respawns than this in one lap -> DNF
    gate_miss_penalty_s: float = 2.0  # slalom pillar passed on the wrong side
    steer: bool = True  # hybrid controller: heading-hold target follows the route
    steer_lookahead: float = 6.0  # mm, pure-pursuit look-ahead along the route
    after_finish: str = "stop"  # stop | loop
    path: str = ""

    @classmethod
    def from_dict(cls, d: dict, path: str = "") -> "CourseSpec":
        d = dict(d)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"course {d.get('name', path)!r}: unknown keys {sorted(unknown)}")
        if "name" not in d:
            d["name"] = Path(path).stem if path else "course"
        raw = d.pop("sections", None)
        if not raw:
            raise ValueError(f"course {d['name']!r} has no sections")
        secs = []
        for i, s in enumerate(raw):
            s = dict(s)
            t = s.get("type")
            if t not in SECTION_DEFAULTS:
                raise ValueError(f"course {d['name']!r} section {i}: unknown type {t!r}; "
                                 f"valid: {sorted(SECTION_DEFAULTS)}")
            bad = set(s) - {"type", "length", "params", "name"}
            if bad:
                raise ValueError(f"section {i} ({t}): unknown keys {sorted(bad)}")
            params = dict(s.get("params") or {})
            bad = set(params) - set(SECTION_DEFAULTS[t])
            if bad:
                raise ValueError(f"section {i} ({t}): unknown params {sorted(bad)}; "
                                 f"valid: {sorted(SECTION_DEFAULTS[t])}")
            length = float(s.get("length", DEFAULT_LENGTH.get(t, 0.0)))
            if length < 0:
                raise ValueError(f"section {i} ({t}): negative length")
            secs.append(SectionSpec(t, length, params, s.get("name", "")))
        if sum(s.type == "finish" for s in secs) > 1:
            raise ValueError("at most one finish section")
        if any(s.type == "finish" for s in secs[:-1]):
            raise ValueError("finish must be the last section")
        if secs[-1].type != "finish":
            secs.append(SectionSpec("finish"))
        spec = cls(sections=secs, path=path, **d)
        if spec.after_finish not in ("stop", "loop"):
            raise ValueError("after_finish must be 'stop' or 'loop'")
        if spec.start_x >= spec.lead_in:
            raise ValueError("start_x must be inside the lead-in (start_x < lead_in)")
        return spec

    @classmethod
    def load(cls, name_or_path: str | Path) -> "CourseSpec":
        """Built-in name (courses/<name>.json|.toml) or a file path."""
        p = Path(name_or_path)
        if not p.suffix:
            for ext in (".json", ".toml"):
                cand = COURSES_DIR / f"{name_or_path}{ext}"
                if cand.exists():
                    p = cand
                    break
            else:
                raise FileNotFoundError(f"no built-in course {name_or_path!r}; available: "
                                        f"{builtin_courses()}")
        if p.suffix == ".toml":
            import tomllib

            data = tomllib.loads(p.read_text())
        else:
            data = json.loads(p.read_text())
        return cls.from_dict(data, path=str(p))


def builtin_courses() -> list[str]:
    return sorted({p.stem for p in COURSES_DIR.glob("*.json")} |
                  {p.stem for p in COURSES_DIR.glob("*.toml")})
