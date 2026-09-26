"""Wiring for ASTEROID DODGE: world + fly + rocks + eyes + brain + game.

``AsteroidSession`` is the standalone install/run API (no app needed)::

    from perpetualfly.games import AsteroidSession, GameBrain
    brain = GameBrain("brain").start()          # or GameBrain("none")
    s = AsteroidSession(brain=brain, seed=0)
    while s.game.state != "gameover":
        s.step()                                # one physics chunk + brain + game
    brain.close()

Per chunk (``chunk_steps`` physics steps, 5 ms by default): physics (rocks move in
a pre-step hook, contacts and looming in post-step hooks; loom events go straight
to the brain), then ``brain.update(run time)`` (clock mark, new BrainStates,
smoothed drive, jump request), then the game logic (outcomes, waves, score).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perpetualfly.games.asteroids import AsteroidConfig, AsteroidField, AsteroidGame
from perpetualfly.games.brain_io import GameBrain
from perpetualfly.games.vision import GameVision
from perpetualfly.vision.looming import LoomingConfig


def game_looming_config() -> LoomingConfig:
    """Eyes for the game: the default field of view (15 deg frontal overlap, rear blind
    sector), 2 ms updates while a rock is out, and a 20 ms photoreceptor/EMD low-pass
    (the rocks are slow; the 1 ms default is tuned for the whip's lash and lets the
    tripod gait's head bob through as expansion noise)."""
    return LoomingConfig(update_every_s=2e-3, idle_every_s=2e-2, tau_s=0.02,
                         persist_s=0.06, refresh_s=0.015, history=2000)


class AsteroidSession:
    game_name = "asteroids"

    def __init__(self, brain: GameBrain | None = None, cfg: AsteroidConfig | None = None,
                 *, seed: int = 0, app_cfg=None, chunk_steps: int = 50,
                 looming: LoomingConfig | None = None, jump: bool | None = None) -> None:
        from perpetualfly.actions.base import ActionManager
        from perpetualfly.config import AppConfig
        from perpetualfly.simulation import Simulation

        self.cfg = cfg or AsteroidConfig()
        self.brain = brain or GameBrain("none")
        if jump is not None:
            self.brain.map.jump = bool(jump)
        app_cfg = app_cfg or AppConfig()
        # the brain steers: no heading hold pulling the fly back to +x
        app_cfg.controller.heading_gain = 0.0
        self.app_cfg = app_cfg
        self.field = AsteroidField(self.cfg)
        self.sim = Simulation(app_cfg, world_extensions=[self.field.extension])
        self.field.attach(self.sim)
        self.actions = ActionManager(self.sim)
        self.chunk_steps = int(chunk_steps)
        self.game = AsteroidGame(self.sim, self.field, self.cfg, seed=seed,
                                 on_respawn=self._on_respawn)
        self.vision = GameVision(self.sim, looming or game_looming_config(),
                                           sink=self.brain.on_loom, time_fn=self.game.time)
        for src in self.field.visual_sources():
            self.vision.add_source(src)
        self.vision.attach()
        self.sim.controller.signal_filter = self.brain.signal_filter
        self.jump_times: list[float] = []
        self.sim.reset()
        self.brain.reset(self.game.time())

    # ------------------------------------------------------------------ loop
    def _on_respawn(self) -> None:
        self.brain.reset(self.game.time())

    def run_time(self) -> float:
        return self.game.time()

    def step(self) -> None:
        self.sim.step(self.chunk_steps)
        rt = self.game.time()
        if self.brain.update(rt) and self.game.state != "gameover":
            self.trigger_jump()
        self.game.after_physics()

    def trigger_jump(self) -> None:
        from perpetualfly.actions.jump import Jump

        self.actions.trigger(Jump(mode=self.brain.map.jump_mode), replace=True, source="brain")
        self.jump_times.append(self.game.time())
        self.game._emit("jump", "JUMP (giant fibre)")

    def restart(self, difficulty: str | None = None) -> None:
        self.game.restart(difficulty)
        self.brain.reset(self.game.time())

    def close(self) -> None:
        self.vision.detach()
        self.sim.close()


class ChaseSession:
    """Wiring for FOLLOW THE LEADER: world + fly + leader fly + eyes (LC10a) + brain.

    Same loop as ``AsteroidSession``: ``step()`` = one physics chunk (the leader
    moves in a pre-step hook, the eyes send LC10a events in a post-step hook), then
    ``brain.update``, then the game rules. No jumps (the giant fibre is not driven
    by this game's input and the jump mapping is off)."""

    game_name = "chase"
    # follow camera: closer and a little higher than ASTEROID DODGE's, so the leader
    # (~5-9 mm ahead) and the follower are both large in the frame
    camera = {"distance": 15.0, "elevation": -30.0, "azimuth": 0.0}

    def __init__(self, brain: GameBrain | None = None, cfg=None, *, seed: int = 0,
                 app_cfg=None, chunk_steps: int = 50, response=None) -> None:
        from perpetualfly.config import AppConfig
        from perpetualfly.games.chase import ChaseConfig, ChaseGame, LeaderFly, PursuitVision
        from perpetualfly.simulation import Simulation

        self.cfg = cfg or ChaseConfig()
        self.brain = brain or GameBrain("none")
        self.brain.map.jump = False
        app_cfg = app_cfg or AppConfig()
        app_cfg.controller.heading_gain = 0.0  # the brain steers
        self.app_cfg = app_cfg
        self.leader = LeaderFly(self.cfg, seed=seed + 7)
        self.sim = Simulation(app_cfg, world_extensions=[self.leader.extension])
        self.leader.attach(self.sim)
        self.chunk_steps = int(chunk_steps)
        self.sim.reset()
        self.game = ChaseGame(self.sim, self.leader, self.cfg, seed=seed,
                              on_respawn=self._on_respawn)
        self.vision = PursuitVision(self.sim, self.leader, sink=self.brain.on_pursuit,
                                    time_fn=self.game.time, response=response)
        self.vision.attach()
        self.sim.controller.signal_filter = self.brain.signal_filter
        self.jump_times: list[float] = []
        self.brain.reset(self.game.time())

    def _on_respawn(self) -> None:
        self.brain.reset(self.game.time())

    def run_time(self) -> float:
        return self.game.time()

    def step(self) -> None:
        self.sim.step(self.chunk_steps)
        self.brain.update(self.game.time())
        self.game.after_physics()

    def restart(self, difficulty: str | None = None) -> None:
        self.game.restart(difficulty)
        self.brain.reset(self.game.time())

    def close(self) -> None:
        self.vision.detach()
        self.leader.detach()
        self.sim.close()


def make_renderer(sim, width: int = 960, height: int = 640, distance: float = 24.0,
                  elevation: float = -24.0, azimuth: float = -15.0):
    """Follow camera behind the fly, high enough to see the rocks coming."""
    from perpetualfly.rendering import FrameRenderer

    cfg = sim.cfg
    return FrameRenderer(sim.model, replace(cfg.render, width=width, height=height),
                         replace(cfg.camera, distance=distance, elevation=elevation,
                                 follow_azimuth_offset=azimuth, heading_tau_s=0.35,
                                 max_distance=max(distance, 30.0)))


def render_frame(renderer, sim) -> np.ndarray:
    return renderer.render(sim.data, sim.time, sim.thorax_position(), sim.heading(),
                           ground_z=0.0, tilt_deg=sim.tilt_deg())
