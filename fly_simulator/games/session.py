"""Wiring for ASTEROID DODGE: world + fly + rocks + eyes + brain + game.

``AsteroidSession`` is the standalone install/run API (no app needed)::

    from fly_simulator.games import AsteroidSession, GameBrain
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

from fly_simulator.games.asteroids import AsteroidConfig, AsteroidField, AsteroidGame
from fly_simulator.games.brain_io import GameBrain
from fly_simulator.games.vision import GameVision
from fly_simulator.vision.looming import LoomingConfig


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
        from fly_simulator.actions.base import ActionManager
        from fly_simulator.config import AppConfig
        from fly_simulator.simulation import Simulation

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
        from fly_simulator.actions.jump import Jump

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
        from fly_simulator.config import AppConfig
        from fly_simulator.games.chase import ChaseConfig, ChaseGame, LeaderFly, PursuitVision
        from fly_simulator.simulation import Simulation

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


class RingsSession:
    """Wiring for FLY THROUGH RINGS: flight world + flying fly + hoops + eyes (LC10a)
    + brain (docs/GAMES.md, game 3).

    The fly is a ``FlightSimulation`` (flapping wings in MuJoCo's fluid model, dt
    5e-5 s) with ``FlightMode`` (take-off, forward flight, crash detection) and an
    ``ActionManager`` (the take-off jump). ``step()`` = one physics chunk
    (``chunk_steps`` x 5e-5 s = 5 ms; the eyes send LC10a events in a post-step
    hook), then ``brain.update``, then the heading command (DNa01/02 -> heading
    rate on ``FlightMode``'s heading goal), then the game rules."""

    game_name = "rings"
    # chase camera behind and a little above the flying fly (the rings ahead in view)
    camera = {"distance": 16.0, "elevation": -14.0, "azimuth": 0.0}

    def __init__(self, brain: GameBrain | None = None, cfg=None, *, seed: int = 0,
                 app_cfg=None, chunk_steps: int = 100, response=None) -> None:
        from fly_simulator.actions.base import ActionManager
        from fly_simulator.config import AppConfig, FlightModeConfig
        from fly_simulator.flight import FlightSimulation
        from fly_simulator.flight.mode import FlightMode
        from fly_simulator.games.rings import (
            FlightPilot, RingCourse, RingsConfig, RingsGame, RingVision, turn_command)

        self._turn_command = turn_command
        self.cfg = cfg or RingsConfig()
        self.brain = brain or GameBrain("none")
        self.brain.map.jump = False
        self.app_cfg = app_cfg or AppConfig()
        self.course = RingCourse(self.cfg, seed=seed + 11)
        self.sim = FlightSimulation(self.app_cfg, world_extensions=[self.course.extension])
        self.course.attach(self.sim)
        self.actions = ActionManager(self.sim)
        self.flight = FlightMode(self.sim, self.actions,
                                 FlightModeConfig(enabled=True, hover_s=None))
        self.pilot = FlightPilot(self.sim, self.flight, self.cfg)
        self.chunk_steps = int(chunk_steps)
        self.jump_times: list[float] = []
        self.unstable = 0
        self._crash_count = 0
        self.sim.reset()
        self.game = RingsGame(self.sim, self.course, self.pilot, self.cfg, seed=seed,
                              on_respawn=self._on_respawn)
        self.vision = RingVision(self.sim, self.course, self.pilot.true_yaw,
                                 sink=self.brain.on_pursuit, time_fn=self.game.time,
                                 response=response)
        self.vision.attach()
        self._last_rt = self.game.time()
        self.brain.reset(self.game.time())

    def _on_respawn(self) -> None:
        self.brain.reset(self.game.time())

    def run_time(self) -> float:
        return self.game.time()

    def turn(self, run_time: float) -> float:
        """DNa01/02 left-right command in [-1, 1] from the latest (fresh) brain state."""
        b = self.brain
        if not b.connected or b.latest is None or b.latest.sim_time is None:
            return 0.0
        if run_time - float(b.latest.sim_time) > b.map.stale_after_s:
            return 0.0
        return self._turn_command(b.rates, self.cfg.r_ref_hz)

    def step(self) -> None:
        from fly_simulator.simulation import SimulationInstabilityError

        crashed = False
        try:
            self.sim.step(self.chunk_steps)
        except SimulationInstabilityError:
            # the model blew up (it happens after hard crashes): count it as a crash
            self.unstable += 1
            crashed = True
            self.game._reset_fly()
        if self.flight.counts.get("crash", 0) != self._crash_count:
            self._crash_count = self.flight.counts.get("crash", 0)
            crashed = True
        rt = self.game.time()
        self.brain.update(rt)
        dt = max(0.0, rt - self._last_rt)
        self._last_rt = rt
        self.pilot.update(self.turn(rt), dt)
        self.game.after_physics(crashed)

    def restart(self, difficulty: str | None = None) -> None:
        self.game.restart(difficulty)
        self._crash_count = self.flight.counts.get("crash", 0)
        self._last_rt = self.game.time()
        self.brain.reset(self.game.time())

    def render(self, renderer) -> np.ndarray:
        """Frame for the HUD: the camera follows the flying fly without zooming out
        for its altitude (``ground_z`` just under it) or freezing its heading for the
        hover posture's ~48 deg pitch."""
        sim = self.sim
        p = sim.thorax_position()
        flying = self.pilot.airborne
        return renderer.render(sim.data, sim.time, p,
                               self.pilot.true_yaw() if flying else sim.heading(),
                               ground_z=float(p[2]) - 2.0 if flying else 0.0,
                               tilt_deg=0.0 if flying else sim.tilt_deg())

    def close(self) -> None:
        self.vision.detach()
        self.sim.close()


class CanyonSession(RingsSession):
    """Wiring for CANYON RUN: flight world + flying fly + pillars + eyes (LC4 / LPLC2)
    + brain (docs/GAMES.md, game 5).

    Same flight stack and loop as ``RingsSession`` (``FlightSimulation`` +
    ``FlightMode`` + ``FlightPilot``; ``step()`` = one 5 ms physics chunk, then
    ``brain.update``, then DNa01/02 -> heading rate, then the rules). The eyes are
    ``CanyonVision`` (ASTEROID DODGE's ``GameVision``: every pillar is a looming
    source) and the sink is ``brain.on_loom``. The game starts with an air start."""

    game_name = "canyon"
    # chase camera behind and above the fly, high enough to look over the pillars
    camera = {"distance": 22.0, "elevation": -27.0, "azimuth": 0.0}

    def __init__(self, brain: GameBrain | None = None, cfg=None, *, seed: int = 0,
                 app_cfg=None, chunk_steps: int = 100, looming: LoomingConfig | None = None,
                 response=None) -> None:
        from fly_simulator.actions.base import ActionManager
        from fly_simulator.config import AppConfig, FlightModeConfig
        from fly_simulator.flight import FlightSimulation
        from fly_simulator.flight.mode import FlightMode
        from fly_simulator.games.canyon import (
            CanyonConfig, CanyonField, CanyonGame, CanyonVision)
        from fly_simulator.games.rings import FlightPilot, turn_command

        self._turn_command = turn_command
        self.cfg = cfg or CanyonConfig()
        self.brain = brain or GameBrain("none")
        self.brain.map.jump = False
        self.app_cfg = app_cfg or AppConfig()
        self.field = CanyonField(self.cfg, seed=seed + 17)
        self.sim = FlightSimulation(self.app_cfg, world_extensions=[self.field.extension])
        self.field.attach(self.sim)
        self.actions = ActionManager(self.sim)
        self.flight = FlightMode(self.sim, self.actions,
                                 FlightModeConfig(enabled=True, hover_s=None))
        self.pilot = FlightPilot(self.sim, self.flight, self.cfg)
        self.chunk_steps = int(chunk_steps)
        self.jump_times: list[float] = []
        self.unstable = 0
        self._crash_count = 0
        self.sim.reset()
        self.game = CanyonGame(self.sim, self.field, self.pilot, self.cfg, seed=seed,
                               on_respawn=self._on_respawn)
        self.vision = CanyonVision(self.sim, looming or game_looming_config(),
                                   self.pilot.true_yaw, sink=self.brain.on_loom,
                                   time_fn=self.game.time)
        for src in self.field.visual_sources(response):
            self.vision.add_source(src)
        sim = self.sim
        self.field.eye_z_fn = lambda: float(sim.thorax_position()[2])
        self.vision.attach()
        self._last_rt = self.game.time()
        self.brain.reset(self.game.time())

    def close(self) -> None:
        self.vision.detach()
        self.field.detach()
        self.sim.close()


class PongSession:
    """Wiring for FLY PONG: world + standing fly on a paddle sled + court + eyes
    (LC10a) + brain (docs/GAMES.md, game 4).

    The fly holds the standing pose (the CPG controller is replaced by the
    standing action, all tarsi adhering); the court's pre-step hook moves the ball
    and the paddles and carries the fly with its sled. ``step()`` = one physics
    chunk (the eyes send LC10a events in a post-step hook), then ``brain.update``,
    then the paddle command (DNa01/02 -> lateral paddle velocity), then the rules.
    ``none``: the paddle command is 0 (the paddle stays where it is)."""

    game_name = "pong"
    # a court camera well behind and above the fly (see ``render``)
    camera = {"distance": 33.0, "elevation": -42.0, "azimuth": 0.0}

    def __init__(self, brain: GameBrain | None = None, cfg=None, *, seed: int = 0,
                 app_cfg=None, chunk_steps: int = 50, response=None) -> None:
        from fly_simulator.config import AppConfig
        from fly_simulator.games.chase import PursuitVision
        from fly_simulator.games.pong import PongConfig, PongCourt, PongGame, paddle_command
        from fly_simulator.simulation import Simulation

        self._paddle_command = paddle_command
        self.cfg = cfg or PongConfig()
        self.brain = brain or GameBrain("none")
        self.brain.map.jump = False
        app_cfg = app_cfg or AppConfig()
        app_cfg.controller.heading_gain = 0.0
        self.app_cfg = app_cfg
        self.court = PongCourt(self.cfg, seed=seed + 13)
        self.sim = Simulation(app_cfg, world_extensions=[self.court.extension])
        ctrl = self.sim.controller
        stand = ctrl.initial_action()  # standing pose, all tarsi adhering

        def _stand():
            ctrl.apply(stand)
            return stand

        ctrl.step_and_apply = _stand
        self.court.attach(self.sim)
        self.chunk_steps = int(chunk_steps)
        self.sim.reset()
        self.game = PongGame(self.sim, self.court, self.cfg, seed=seed,
                             on_respawn=self._on_respawn)
        self.vision = PursuitVision(self.sim, self.court, sink=self.brain.on_pursuit,
                                    time_fn=self.game.time, response=response)
        self.vision.attach()
        self.jump_times: list[float] = []
        self.turn = 0.0
        self.brain.reset(self.game.time())

    def _on_respawn(self) -> None:
        self.brain.reset(self.game.time())

    def run_time(self) -> float:
        return self.game.time()

    def turn_now(self, run_time: float) -> float:
        """DNa01/02 left-right command in [-1, 1] from the latest (fresh) brain state."""
        from fly_simulator.games.rings import turn_command

        b = self.brain
        if not b.connected or b.latest is None or b.latest.sim_time is None:
            return 0.0
        if run_time - float(b.latest.sim_time) > b.map.stale_after_s:
            return 0.0
        return turn_command(b.rates, self.cfg.r_ref_hz)

    def step(self) -> None:
        self.sim.step(self.chunk_steps)
        rt = self.game.time()
        self.brain.update(rt)
        self.turn = self.turn_now(rt)
        self.court.phys.pcmd = self.cfg.paddle_vmax * self.turn
        self.game.after_physics()

    def restart(self, difficulty: str | None = None) -> None:
        self.game.restart(difficulty)
        self.brain.reset(self.game.time())

    def render(self, renderer) -> np.ndarray:
        """A court camera behind the fly: it looks down the court and follows the
        paddle only partly, so the whole court stays in view."""
        c = self.cfg
        p = self.sim.thorax_position()
        target = np.array([c.paddle_x + 0.26 * c.court_len, 0.35 * float(p[1]), 0.0])
        return renderer.render(self.sim.data, self.sim.time, target, 0.0, ground_z=0.0,
                               tilt_deg=0.0)

    def close(self) -> None:
        self.vision.detach()
        self.court.detach()
        self.sim.close()


def make_renderer(sim, width: int = 960, height: int = 640, distance: float = 24.0,
                  elevation: float = -24.0, azimuth: float = -15.0):
    """Follow camera behind the fly, high enough to see the rocks coming."""
    from fly_simulator.rendering import FrameRenderer

    cfg = sim.cfg
    return FrameRenderer(sim.model, replace(cfg.render, width=width, height=height),
                         replace(cfg.camera, distance=distance, elevation=elevation,
                                 follow_azimuth_offset=azimuth, heading_tau_s=0.35,
                                 max_distance=max(distance, 30.0)))


def render_frame(renderer, sim) -> np.ndarray:
    return renderer.render(sim.data, sim.time, sim.thorax_position(), sim.heading(),
                           ground_z=0.0, tilt_deg=sim.tilt_deg())
