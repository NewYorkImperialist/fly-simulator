"""DELIVERY PILOT: the fly delivers packages by air forever (docs/JOBS.md, "delivery_pilot").

The first job on the **real flight fly** (``needs_flight = True``: the flight body with
flapping wings in MuJoCo's fluid model, dt 5e-5 s, docs/FLIGHT.md). Every newton of
lift, thrust and steering comes from the beating wings; no external force acts on the
fly at any point.

* **Scene** (fly scale, 1 mm ~ 0.7 m): a sorting depot with a flat roof (loading mark,
  a parcel shelf, a windsock) at the origin and ``houses`` (flat roof terraces with a
  doormat, a mailbox, a porch lamp and a house number on the attic), street, lawns and
  trees. Roofs / walls / attics collide (static); the rest is decoration.
* **Loop**: on the depot roof a parcel slides from the shelf under the fly (``LOAD``),
  the fly takes off with ``FlightMode``'s real take-off (the escape ``Jump`` -> wings at
  the end of the leg stroke -> ``HoverController``), climbs to the cruise altitude,
  turns toward the address, flies there (waypoint guidance through the flight
  controller's inputs: heading goal, forward speed, altitude set point; cruise in
  velocity control along the bearing to the doormat with a trapezoid speed profile,
  then the position loop over the doormat with a slow set-point integrator that leans
  into the wind), descends over the roof and lands with ``FlightMode``'s landing sequence (descend at
  15 mm/s, adhesion at the first leg contact, pitch-down, wings off). It drops the
  parcel on the doormat (DELIVERED, *ding-dong*, the mailbox flag goes up and the porch
  lamp lights), takes off again, flies back to the depot, lands, loads the next parcel,
  and so on forever. Addresses cycle 1, 2, 3, 4.
* **Failures are counted, never hidden**: a landing that ends off the target roof (a
  bounce, a slide off the edge, the street) is a *missed landing*: the fly takes off
  again from where it is and retries. A crash (``FlightMode`` crash: body contact or
  tilt > 120 deg while airborne) is a *crash landing*; if the fly then lies down for
  ``recover_after_s`` the base class makes an explicit, counted reset (respawn on the
  depot roof; the parcel goes back to the shelf and the order stays open).
* **Engineered, labelled**: the parcel is carried *kinematically* (a mocap body hung
  0.55 mm under the fly's centre of mass on a drawn string; visual, no mass, no
  contacts), so it does not load the flight controller; a real payload would need the
  controller's mass / inertia terms updated. The guidance (speed profile, gains, when
  to land) is our code, not the brain. The **wind** is physical: MuJoCo's medium
  velocity (``model.opt.wind``), a slow gentle breeze (``wind_max`` mm/s) that the
  fluid model applies to the wings and body drag, shown by the windsock; the flight
  controller rejects it with its position / altitude integrators. The DELIVERED caption
  is a screen-space overlay (``post_process``), not part of the scene.

Counters: parcels delivered (the work counter), flight time, distance flown (COM path
while the wings are on), on-time % (a delivery is on time when it lands within
``budget_s + distance / budget_speed`` of the pickup), missed landings, crash landings,
recoveries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import cv2
import mujoco as mj
import numpy as np

from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, contact_kwargs, quat_axis_angle, wrap_angle
from fly_simulator.jobs.kebab_assets import add_mesh, add_texture, add_textured_material, fbm
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

P = "deliv/"

# (x, y, roof top z, house number): flat roof terraces, all in mm
DEFAULT_HOUSES = ((40.0, 16.0, 5.0, 1), (30.0, -34.0, 6.5, 2), (-18.0, 40.0, 4.0, 3),
                  (-38.0, -22.0, 5.5, 4))


@dataclass
class DeliveryConfig(JobConfig):
    update_every_steps: int = 10  # 0.5 ms at the flight timestep
    stuck_timeout_s: float = 90.0  # no delivery this long -> explicit reset
    # --- the town (mm) -------------------------------------------------------------
    depot_top: float = 3.0  # depot roof height (the fly spawns on it)
    depot_half: float = 7.0  # half size of the depot roof (square)
    houses: tuple = DEFAULT_HOUSES
    house_half: float = 5.5  # half size of a house's roof terrace (square)
    attic_depth: float = 2.2  # the attic block on the far side of each roof (collides)
    mat_forward: float = 1.5  # the doormat / landing point: this far from the roof centre
    # toward the depot (away from the attic)
    land_lead: float = 1.0  # aim this much further toward the depot (landings slide forward)
    # --- flight guidance (our code; the flight itself is FlightMode + HoverController) ---
    cruise_alt: float = 12.0  # COM height above the street while cruising
    min_alt_margin: float = 2.5  # start forward flight when this far above every roof
    cruise_speed: float = 120.0  # mm/s
    accel: float = 400.0  # mm/s^2, speed command ramp up
    decel: float = 250.0  # mm/s^2, speed command ramp down before the address
    turn_first_deg: float = 20.0  # heading error above this: turn before moving on
    turn_rate: float = 3.0  # rad/s, heading slew in the air (FlightMode.turn_rate)
    descend_at_mm: float = 10.0  # this close to the doormat: descend
    approach_clear: float = 2.5  # COM height above the roof before the landing starts
    cruise_xy_zeta: float = 3.0  # velocity-loop damping while cruising (FlightMode escape: 3)
    handover_mm: float = 1.5  # velocity -> position control this far from the doormat
    approach_xy_wn: float = 7.0  # position loop over the doormat (HoverGains: 5)
    approach_xy_zeta: float = 1.2
    approach_xy_ki: float = 30.0
    lean_ki: float = 1.5  # 1/s: set point leans into the wind (outer integrator)
    lean_max: float = 4.0  # mm
    lean_zone_mm: float = 2.0  # the lean integrates only this close to the landing point
    land_radius: float = 1.2  # mm, COM over the landing point before landing
    land_max_speed: float = 20.0  # mm/s horizontal
    approach_timeout_s: float = 2.0  # still not settled over the mat: land anyway
    leg_timeout_s: float = 12.0  # take-off -> touchdown longer than this: land now
    # --- ground work ------------------------------------------------------------------
    load_s: float = 0.6  # parcel slides onto the fly
    drop_s: float = 0.6  # parcel onto the doormat, ding-dong
    settle_s: float = 0.25  # after a touchdown before judging the landing
    retry_after_s: float = 0.6  # missed landing / crash: upright this long -> take off again
    max_retry_tilt_deg: float = 30.0
    carry_dz: float = -1.7  # parcel centre below the COM in the air (kinematic, labelled)
    parcel_half: float = 0.38
    # --- on time ------------------------------------------------------------------------
    budget_s: float = 5.0  # + distance / budget_speed = the delivery window
    budget_speed: float = 40.0
    # --- wind (physical: MuJoCo medium velocity) ------------------------------------------
    wind_max: float = 25.0  # mm/s (0 = still air)
    wind_period_s: float = 23.0
    # --- presentation ---------------------------------------------------------------------
    caption_s: float = 1.4
    cam_distance: float = 22.0
    cam_elevation: float = -20.0
    cam_yaw_tau_s: float = 1.2
    shadows: bool = False  # sun shadows (off: the town-wide shadow map is coarse)


# ---------------------------------------------------------------------------
# textures (generated in code, nothing on disk)
# ---------------------------------------------------------------------------


def _street_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Asphalt: a smooth grey tone with faint paving joints (low detail on purpose:
    fine grit shimmers in a moving camera and bloats GIFs)."""
    rng = np.random.default_rng(seed + 5)
    f = fbm(n, n, 4, 4, rng, octaves=3)
    g = (0.40 + 0.05 * f)[..., None] * np.array([1.0, 1.0, 1.03], np.float32)
    img = np.clip(g * 255, 0, 255).astype(np.uint8)
    for k in (0, n // 2):
        cv2.line(img, (k, 0), (k, n - 1), (92, 92, 95), 1)
        cv2.line(img, (0, k), (n - 1, k), (92, 92, 95), 1)
    return img


def _roof_texture(rgb, seed: int = 0, n: int = 128, tiles: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    f = fbm(n, n, 4, 4, rng, octaves=3)
    base = np.array(rgb, np.float32)[None, None, :] * (0.88 + 0.2 * f[..., None])
    img = np.clip(base * 255, 0, 255).astype(np.uint8)
    step = n // tiles
    dark = tuple(int(v * 0.72 * 255) for v in rgb)
    for k in range(0, n, step):
        cv2.line(img, (k, 0), (k, n - 1), dark, 1)
        cv2.line(img, (0, k), (n - 1, k), dark, 1)
    return img


def _label_texture(text: str, bg, ink, h: int = 96, w: int = 256, scale: float = 1.6,
                   border: bool = True) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = [int(255 * c) for c in bg]
    if border:
        cv2.rectangle(img, (4, 4), (w - 5, h - 5), [int(255 * c) for c in ink], 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), font, scale,
                [int(255 * c) for c in ink], 3, cv2.LINE_AA)
    return img


def _mat_texture(n: int = 64) -> np.ndarray:
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = (120, 78, 40)
    for k in range(3, n, 5):
        cv2.line(img, (k, 3), (k, n - 4), (95, 60, 30), 1)
    cv2.rectangle(img, (1, 1), (n - 2, n - 2), (70, 44, 22), 2)
    return img


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@register_job
class DeliveryPilotJob(EternalJob):
    name = "delivery_pilot"
    #: depth range for a town-sized scene (the job camera is >= ~8 mm away);
    #: applied by EternalJob.attach
    znear = 0.3
    zfar = 400.0
    title = "DELIVERY PILOT"
    tagline = "the fly delivers packages by air forever"
    work_label = "parcels delivered"
    config_cls = DeliveryConfig
    needs_flight = True
    required_names = (P + "depot_roof", P + "parcel0", P + "house1_roof")

    def __init__(self, cfg: DeliveryConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.houses = [tuple(float(v) for v in h[:3]) + (int(h[3]),) for h in c.houses]
        self.n_parcels = len(self.houses) + 2
        self.phase = "load"
        self.t_phase = 0.0
        self.dest = -1  # -1 = the depot, else a house index
        self.order = 0  # the house the current parcel is for
        self.next_house = 0
        self.carrying: int | None = None  # parcel index
        self.parcel_state: list = []  # "shelf" / "carried" / ("mat", house)
        self.parcel_anim: list = []  # (t0, from xyz, to xyz) or None
        self.t_pickup = 0.0
        self._reopened = False
        self.order_dist = 0.0
        # counters
        self.n_delivered = 0
        self.n_on_time = 0
        self.n_missed = 0
        self.n_crash = 0
        self.n_takeoffs = 0
        self.n_landings = 0
        self.n_flight_aborts = 0
        self.per_house = [0] * len(self.houses)
        self.flight_time = 0.0
        self.distance = 0.0
        self.best_leg_s = float("inf")
        self.last_leg_s = 0.0
        self.caption = ""
        self.caption_sub = ""
        self._caption_until = -1.0
        self._caption_rgb = (255, 255, 255)
        # guidance state
        self._v = 0.0
        self._t_leg0 = None
        self._last_xy = None
        self._t_dist = 0.0
        self._cam_yaw = 0.0
        self._cam_t = None
        self._crashed_at = None
        self._wind = np.zeros(2)
        self._upright_since = None
        self._lean = np.zeros(2)
        self._touched = False

    # ------------------------------------------------------------ build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)  # flat, no auto hits, flight fly (needs_flight)
        c = self.cfg
        app_cfg.fly.spawn_height = c.depot_top + 0.8
        f = app_cfg.flight
        f.hover_s = None  # the job decides when to land
        f.turn_rate = c.turn_rate

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._materials(spec)
        grid = spec.material("grid")
        if grid is not None:  # the street: asphalt, one texture tile per 24 mm
            grid.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_street"
            grid.texrepeat = [2000.0 / 24.0, 2000.0 / 24.0]
            grid.rgba = (1.0, 1.0, 1.0, 1.0)
            grid.reflectance = 0.0
        # --- the depot: a flat roof on the origin (the fly spawns here) ----------------
        h, t = c.depot_half, c.depot_top
        add_box(wb, P + "depot_roof", (h, h, t / 2), (0.0, 0.0, t / 2), material=P + "depot",
                collide="static")
        add_box(wb, P + "depot_mark", (1.6, 1.6, 0.004), (0.0, 0.0, t + 0.004),
                material=P + "loadmark", **vis)
        # the parcel shelf behind the loading mark (-x side)
        sx = -h + 1.2
        add_box(wb, P + "shelf", (0.7, 3.0, 0.05), (sx, 0.0, t + 0.9), material=P + "wood", **vis)
        for k, yy in enumerate((-2.8, 2.8)):
            add_box(wb, P + f"shelf_leg{k}", (0.06, 0.06, 0.45), (sx, yy, t + 0.45),
                    material=P + "wood", **vis)
        for k, (dx, dy, dz) in enumerate(((-0.3, -1.6, 0), (0.2, 1.8, 0), (0.0, 0.9, 0.6))):
            add_box(wb, P + f"decor_parcel{k}", (0.4, 0.4, 0.3),
                    (sx + dx - 1.4, dy, t + 0.3 + dz), material=P + "cardboard", **vis)
        # the depot sign (faces +x, toward the loading mark) and a hangar front
        A = front_panel_mesh(3.4, 0.9, 0.08)
        add_mesh(spec, P + "sign_mesh", A)
        for k, yaw in enumerate((0.0, math.pi)):  # readable from both sides
            wb.add_geom(name=P + ("depot_sign" if k == 0 else "depot_sign_back"),
                        type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sign_mesh",
                        pos=(-h + 0.3 + (0.05 if k == 0 else -0.05), 0.0, t + 2.9),
                        quat=quat_axis_angle((0, 0, 1), yaw), material=P + "sign_depot", **vis)
        for s in (-1, 1):
            add_box(wb, P + f"sign_post{s:+d}", (0.07, 0.07, 1.0), (-h + 0.3, s * 3.0, t + 1.0),
                    material=P + "steel", **vis)
        # the windsock (a pole + a cone on a mocap body that points with the wind)
        wx, wy = h - 1.0, h - 1.0
        add_box(wb, P + "sock_pole", (0.05, 0.05, 1.6), (wx, wy, t + 1.6), material=P + "steel",
                **vis)
        sock = wb.add_body(name=P + "windsock", mocap=True, pos=(wx, wy, t + 3.0))
        for k in range(4):
            sock.add_geom(name=P + f"sock{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(0.28 - 0.045 * k, 0.2, 0), pos=(0.2 + 0.4 * k, 0, 0),
                          quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                          material=P + ("sock_r" if k % 2 == 0 else "sock_w"), **vis)
        # --- the houses ----------------------------------------------------------------------
        hh = c.house_half
        for i, (x, y, top, num) in enumerate(self.houses):
            q = P + f"house{num}_"
            face = math.atan2(-y, -x)  # the doormat side faces the depot
            R = np.array([[math.cos(face), -math.sin(face)], [math.sin(face), math.cos(face)]])
            quat = quat_axis_angle((0, 0, 1), face)
            # walls + the roof terrace slab (both collide; separate so each has its look)
            add_box(wb, q + "walls", (hh, hh, (top - 0.06) / 2), (x, y, (top - 0.06) / 2),
                    quat=quat, material=P + f"wall{i % 2}", collide="static")
            add_box(wb, q + "roof", (hh + 0.06, hh + 0.06, 0.03), (x, y, top - 0.03), quat=quat,
                    material=P + f"roof{i % 2}", collide="static")
            # walls with windows (visual strips just outside the walls)
            for side in range(4):
                a = face + side * math.pi / 2
                n = np.array([math.cos(a), math.sin(a)])
                for k, zz in enumerate((top * 0.35, top * 0.72)):
                    p = np.array([x, y]) + n * (hh + 0.08)
                    add_box(wb, q + f"win{side}_{k}", (0.01, hh * 0.55, top * 0.1),
                            (p[0], p[1], zz), quat=quat_axis_angle((0, 0, 1), a),
                            material=P + "window", **vis)
            # the attic on the far side (collides: fly into it = a crash)
            ad = c.attic_depth
            back = np.array([x, y]) + R @ np.array([-(hh - ad / 2), 0.0])
            ah = 2.2
            add_box(wb, q + "attic", (ad / 2, hh, ah / 2), (back[0], back[1], top + ah / 2),
                    quat=quat, material=P + f"wall{i % 2}", collide="static")
            # pitched roof on the attic (visual: two slanted slabs)
            for s in (-1, 1):
                pitch = 0.55
                ridge = quat_axis_angle((0, 1, 0), s * pitch)
                off = back + R @ np.array([s * ad / 4, 0.0])
                add_box(wb, q + f"tiles{s:+d}", (ad / 4 / math.cos(pitch) + 0.12, hh + 0.15, 0.06),
                        (off[0], off[1], top + ah + ad / 4 * math.tan(pitch) + 0.04),
                        quat=_qmul(quat, ridge), material=P + "tiles", **vis)
            # house number on the attic front (faces the doormat / the depot)
            num_p = back + R @ np.array([ad / 2 + 0.02, 0.0])
            add_mesh(spec, q + "num_mesh", front_panel_mesh(0.9, 0.7, 0.03))
            wb.add_geom(name=q + "number", type=mj.mjtGeom.mjGEOM_MESH, meshname=q + "num_mesh",
                        pos=(num_p[0], num_p[1], top + 1.1), quat=quat,
                        material=P + f"num{num}", **vis)
            # doormat (landing point), mailbox, porch lamp
            mat_p = np.array([x, y]) + R @ np.array([c.mat_forward, 0.0])
            add_box(wb, q + "doormat", (1.3, 1.0, 0.006), (mat_p[0], mat_p[1], top + 0.012),
                    quat=quat, material=P + "doormat", **vis)
            mb = np.array([x, y]) + R @ np.array([hh * 0.35, hh * 0.72])
            add_box(wb, q + "mailbox_post", (0.06, 0.06, 0.5), (mb[0], mb[1], top + 0.5),
                    material=P + "steel", **vis)
            add_box(wb, q + "mailbox", (0.45, 0.28, 0.3), (mb[0], mb[1], top + 1.2), quat=quat,
                    material=P + "mailbox", **vis)
            flag = wb.add_body(name=q + "flag", mocap=True, pos=(mb[0], mb[1], top + 1.2),
                               quat=quat)
            flag.add_geom(name=q + "flag_bar", type=mj.mjtGeom.mjGEOM_BOX, size=(0.03, 0.03, 0.35),
                          pos=(0.0, -0.31, 0.2), material=P + "flag", **vis)
            flag.add_geom(name=q + "flag_tip", type=mj.mjtGeom.mjGEOM_BOX, size=(0.14, 0.03, 0.1),
                          pos=(0.1, -0.31, 0.45), material=P + "flag", **vis)
            lamp = back + R @ np.array([ad / 2 + 0.05, hh * 0.6])
            wb.add_geom(name=q + "lamp", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.22, 0, 0),
                        pos=(lamp[0], lamp[1], top + 1.6), material=P + "lamp_off", **vis)
            # lawn around the house
            add_box(wb, q + "lawn", (hh + 3.0, hh + 3.0, 0.03), (x, y, 0.03), quat=quat,
                    material=P + "lawn", **vis)
        # --- street decoration: a few trees -------------------------------------------------
        rng = np.random.default_rng(c.seed + 3)
        for k, (tx, ty) in enumerate(((16, 30), (22, -8), (-6, -30), (-30, 10), (8, 52),
                                      (56, -12), (-50, 30), (12, -52))):
            s = rng.uniform(0.8, 1.2)
            wb.add_geom(name=P + f"tree{k}_trunk", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(0.35 * s, 2.5 * s, 0), pos=(tx, ty, 2.5 * s),
                        material=P + "trunk", **vis)
            wb.add_geom(name=P + f"tree{k}_crown", type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=(2.6 * s, 0, 0), pos=(tx, ty, 6.5 * s), material=P + "leaves", **vis)
        # --- the parcel pool (mocap, visual): on the shelf / carried / on a doormat --------
        for k in range(self.n_parcels):
            b = wb.add_body(name=P + f"parcel{k}", mocap=True, pos=self._shelf_slot(k))
            ph = c.parcel_half
            b.add_geom(name=P + f"parcel{k}_box", type=mj.mjtGeom.mjGEOM_BOX,
                       size=(ph, ph, ph * 0.8), material=P + "cardboard", **vis)
            b.add_geom(name=P + f"parcel{k}_tape", type=mj.mjtGeom.mjGEOM_BOX,
                       size=(ph + 0.005, 0.07, ph * 0.8 + 0.005), material=P + "tape", **vis)
            b.add_geom(name=P + f"parcel{k}_string", type=mj.mjtGeom.mjGEOM_CYLINDER,
                       size=(0.015, 0.15, 0), pos=(0, 0, ph * 0.8 + 0.15),
                       rgba=(0.92, 0.92, 0.86, 0.0), **vis)
        # --- light ------------------------------------------------------------------------
        spec.visual.headlight.ambient = (0.34, 0.34, 0.34)
        spec.visual.headlight.diffuse = (0.36, 0.36, 0.36)
        spec.visual.headlight.specular = (0.08, 0.08, 0.08)
        sun = np.array([-40.0, -60.0, 120.0])
        wb.add_light(name=P + "sun", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=tuple(sun),
                     dir=tuple(-sun), diffuse=(0.42, 0.41, 0.37), specular=(0.15, 0.15, 0.15),
                     castshadow=bool(c.shadows))

    def _materials(self, spec) -> None:
        c = self.cfg
        add_texture(spec, P + "tex_street", _street_texture(c.seed))
        add_texture(spec, P + "tex_depot", _roof_texture((0.62, 0.64, 0.66), c.seed, tiles=6))
        add_textured_material(spec, P + "depot", P + "tex_depot", rgba=(1, 1, 1, 1),
                              specular=0.1, texrepeat=(1, 1))
        for i, rgb in enumerate(((0.80, 0.52, 0.36), (0.70, 0.66, 0.52))):
            add_texture(spec, P + f"tex_roof{i}", _roof_texture(rgb, c.seed + i))
            add_textured_material(spec, P + f"roof{i}", P + f"tex_roof{i}", rgba=(1, 1, 1, 1),
                                  specular=0.1)
        for i, rgb in enumerate(((0.93, 0.88, 0.78, 1), (0.78, 0.86, 0.92, 1))):
            spec.add_material(name=P + f"wall{i}", rgba=rgb, specular=0.1)
        add_texture(spec, P + "tex_sign", _label_texture("PARCEL DEPOT", (0.12, 0.28, 0.55),
                                                         (1.0, 0.85, 0.2), w=384, scale=1.8))
        add_textured_material(spec, P + "sign_depot", P + "tex_sign", rgba=(1, 1, 1, 1),
                              emission=0.3)
        for _, _, _, num in self.houses:
            add_texture(spec, P + f"tex_num{num}", _label_texture(str(num), (0.97, 0.96, 0.9),
                                                                  (0.1, 0.12, 0.2), 96, 128, 2.4))
            add_textured_material(spec, P + f"num{num}", P + f"tex_num{num}", rgba=(1, 1, 1, 1),
                                  emission=0.2)
        add_texture(spec, P + "tex_mat", _mat_texture())
        add_textured_material(spec, P + "doormat", P + "tex_mat", rgba=(1, 1, 1, 1))
        for name, rgba, kw in (
                ("loadmark", (1.0, 0.8, 0.1, 1), dict(emission=0.2)),
                ("wood", (0.55, 0.38, 0.22, 1), {}),
                ("cardboard", (0.86, 0.66, 0.40, 1), dict(specular=0.05)),
                ("tape", (0.2, 0.45, 0.85, 1), {}),
                ("string", (0.9, 0.9, 0.85, 1), {}),
                ("steel", (0.6, 0.62, 0.65, 1), dict(specular=0.6)),
                ("sock_r", (0.95, 0.35, 0.1, 1), dict(emission=0.1)),
                ("sock_w", (0.97, 0.97, 0.95, 1), {}),
                ("window", (0.25, 0.4, 0.55, 1), dict(specular=0.8, shininess=0.8)),
                ("tiles", (0.62, 0.24, 0.18, 1), {}),
                ("mailbox", (0.2, 0.35, 0.7, 1), dict(specular=0.4)),
                ("flag", (0.9, 0.12, 0.1, 1), dict(emission=0.2)),
                ("lamp_off", (0.5, 0.48, 0.4, 1), {}),
                ("lamp_on", (1.0, 0.92, 0.5, 1), dict(emission=1.0)),
                ("lawn", (0.34, 0.56, 0.24, 1), {}),
                ("trunk", (0.42, 0.3, 0.2, 1), {}),
                ("leaves", (0.25, 0.5, 0.22, 1), {})):
            spec.add_material(name=P + name, rgba=rgba, **kw)

    def _shelf_slot(self, k: int) -> tuple[float, float, float]:
        c = self.cfg
        n = self.n_parcels
        y = -2.4 + 4.8 * k / max(n - 1, 1)
        return (-c.depot_half + 1.2, y, c.depot_top + 0.95 + c.parcel_half * 0.8)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        self.fm = self.session.flight
        if self.fm is None:
            raise RuntimeError("delivery_pilot needs the flight fly (cfg.flight.enabled)")
        # altitude = height above the street: the job sets absolute set points
        self.fm.ground_height_fn = lambda x, y: 0.0
        self.fm.listeners.append(self._on_flight_event)
        self.mocap_parcel = [int(m.body_mocapid[m.body(P + f"parcel{k}").id])
                             for k in range(self.n_parcels)]
        self.string_gid = [m.geom(P + f"parcel{k}_string").id for k in range(self.n_parcels)]
        self.mocap_sock = int(m.body_mocapid[m.body(P + "windsock").id])
        self.mocap_flag = [int(m.body_mocapid[m.body(P + f"house{h[3]}_flag").id])
                           for h in self.houses]
        self.flag_quat0 = [self.sim.data.mocap_quat[i].copy() for i in self.mocap_flag]
        self.lamp_gid = [m.geom(P + f"house{h[3]}_lamp").id for h in self.houses]
        self.mat_on = m.material(P + "lamp_on").id
        self.mat_off = m.material(P + "lamp_off").id
        self._lamp_until = [-1.0] * len(self.houses)
        self._flag_up = [False] * len(self.houses)
        self.parcel_state = ["shelf"] * self.n_parcels
        self.parcel_anim = [None] * self.n_parcels
        self._begin_load()

    # ------------------------------------------------------------ geometry helpers
    def pad(self, idx: int) -> tuple[float, float, float, float]:
        """(x, y, top z, half size) of the depot (-1) or a house."""
        c = self.cfg
        if idx < 0:
            return 0.0, 0.0, c.depot_top, c.depot_half
        x, y, top, _ = self.houses[idx]
        return x, y, top, c.house_half

    def landing_point(self, idx: int) -> np.ndarray:
        x, y, _, _ = self.pad(idx)
        if idx < 0:
            return np.array([x, y])
        face = math.atan2(-y, -x)
        # aim a little on the depot side of the mat: a landing slides ~1 mm forward
        f = self.cfg.mat_forward + self.cfg.land_lead
        return np.array([x + f * math.cos(face), y + f * math.sin(face)])

    def _in_pad(self, idx: int, x: float, y: float, margin: float = 0.0) -> bool:
        px, py, _, h = self.pad(idx)
        if idx >= 0:
            face = math.atan2(-py, -px)
            dx, dy = x - px, y - py
            c, s = math.cos(face), math.sin(face)
            u, v = c * dx + s * dy, -s * dx + c * dy
        else:
            u, v = x - px, y - py
        return abs(u) <= h - margin and abs(v) <= h - margin

    def ground_height(self, x: float, y: float) -> float:
        for idx in range(-1, len(self.houses)):
            if self._in_pad(idx, x, y):
                return self.pad(idx)[2]
        return 0.0

    def on_pad(self, idx: int) -> bool:
        p = self.sim.com()
        _, _, top, _ = self.pad(idx)
        return self._in_pad(idx, float(p[0]), float(p[1]), 0.2) and p[2] > top + 0.2

    def dest_name(self, idx: int | None = None) -> str:
        idx = self.dest if idx is None else idx
        return "DEPOT" if idx < 0 else f"No. {self.houses[idx][3]}"

    # ------------------------------------------------------------ phases
    def _set(self, phase: str) -> None:
        self.phase = phase
        self.state = phase
        self.t_phase = self.sim.time

    def _flash(self, text: str, sub: str = "", rgb=(255, 255, 255)) -> None:
        self.caption, self.caption_sub, self._caption_rgb = text, sub, rgb
        self._caption_until = self.run_time() + self.cfg.caption_s

    def _begin_load(self) -> None:
        self.dest = -1
        self.steering.set(None, 0.0)
        self._set("load")

    def _pick_parcel(self) -> None:
        """A parcel from the shelf slides under the fly; the order is for next_house."""
        self.order = self.next_house
        self.next_house = (self.next_house + 1) % len(self.houses)
        # the receiver of this address took its last parcel in: back to the shelf
        for k, st in enumerate(self.parcel_state):
            if st == ("mat", self.order):
                self._place_parcel(k, self._shelf_slot(k))
                self.parcel_state[k] = "shelf"
        k = next((i for i, st in enumerate(self.parcel_state) if st == "shelf"), 0)
        self.carrying = k
        self.parcel_state[k] = "carried"
        self.parcel_anim[k] = (self.sim.time, self.sim.data.mocap_pos[self.mocap_parcel[k]].copy())
        if not self._reopened:  # (after a reset the open order's clock keeps running)
            self.t_pickup = self.run_time()
        self._reopened = False
        self.order_dist = float(np.hypot(*self.landing_point(self.order)))
        self._flash("PARCEL LOADED", f"for {self.dest_name(self.order)}", (255, 220, 120))

    def _takeoff(self) -> None:
        msg = self.fm.takeoff(source="job")
        if msg.startswith("[flight] take-off"):
            self.n_takeoffs += 1
            self._t_leg0 = self.sim.time
            self._v = 0.0
            self._set("takeoff")
        else:
            self._set("wait")

    def _deliver(self) -> None:
        k = self.carrying
        h = self.order
        self.carrying = None
        x, y, top, _ = self.pad(h)
        face = math.atan2(-y, -x)
        # beside the fly on the doormat, toward the depot
        f = self.cfg.mat_forward + 0.5  # the doormat's centre + 0.5 mm toward the street
        tgt = np.array([x + f * math.cos(face), y + f * math.sin(face),
                        top + 0.018 + self.cfg.parcel_half * 0.8])
        if k is not None:
            self.parcel_state[k] = ("mat", h)
            self.parcel_anim[k] = (self.sim.time, self.sim.data.mocap_pos[self.mocap_parcel[k]].copy(),
                                   tgt)
        self.n_delivered += 1
        self.per_house[h] += 1
        took = self.run_time() - self.t_pickup
        window = self.cfg.budget_s + self.order_dist / self.cfg.budget_speed
        on_time = took <= window
        self.n_on_time += int(on_time)
        self.last_leg_s = took
        self.best_leg_s = min(self.best_leg_s, took)
        self.add_work(1)
        self._lamp_until[h] = self.run_time() + 2.5
        self._flag_up[h] = True
        self.say(f"DELIVERED #{self.n_delivered} to {self.dest_name(h)} in {took:.1f} s "
                 f"({'on time' if on_time else 'late'}) - ding-dong!")
        self.session.log_event("delivery", job=self.name, house=self.houses[h][3],
                               took_s=round(took, 2), on_time=on_time)
        self._flash("DELIVERED!", f"No. {self.houses[h][3]}  *ding-dong*   "
                    f"{took:.1f} s {'ON TIME' if on_time else 'LATE'}", (120, 255, 140))

    # ------------------------------------------------------------ events
    def _on_flight_event(self, ev) -> None:
        if ev.kind == "crash":
            self.n_crash += 1
            self._crashed_at = self.sim.time
            self._flash("CRASH LANDING!", "", (255, 110, 90))
            self._set("down")
        elif ev.kind == "abort":
            self.n_flight_aborts += 1
            if self.phase in ("takeoff", "climb", "cruise", "approach", "landing"):
                self._set("down")
        elif ev.kind == "land":
            self.n_landings += 1

    def on_reset(self) -> None:
        # explicit reset: the fly is back on the depot roof; the parcel goes back to
        # the shelf, the order stays open (its clock keeps running)
        if self.carrying is not None:
            k = self.carrying
            self.parcel_state[k] = "shelf"
            self.parcel_anim[k] = None
            self.carrying = None
            self.next_house = self.order
            self._reopened = True
        for k, st in enumerate(self.parcel_state):
            if st in ("shelf", "carried"):
                self.parcel_state[k] = "shelf"
                self._place_parcel(k, self._shelf_slot(k))
        self._v = 0.0
        self._last_xy = None
        self.sim.model.opt.wind[:] = 0.0
        self._begin_load()
        self.t_phase = self.sim.time

    # ------------------------------------------------------------ the loop
    def update(self) -> None:
        sim, fm, c = self.sim, self.fm, self.cfg
        dt = c.update_every_steps * sim.timestep
        t = sim.time - self.t_phase
        self._update_wind()
        airborne = fm.airborne
        if airborne:
            self.flight_time += dt
            if abs(sim.time - self._t_dist) >= 0.01:  # (sim time restarts after a reset)
                self._t_dist = sim.time
                xy = sim.com()[:2]
                if self._last_xy is not None:
                    self.distance += float(np.hypot(*(xy - self._last_xy)))
                self._last_xy = xy
        else:
            self._last_xy = None
        ph = self.phase
        if ph == "load":
            if self.carrying is None and t >= 0.1 and self.on_pad(-1):
                self._pick_parcel()
            if t >= c.load_s and fm.state == "walking" and sim.tilt_deg() < c.max_retry_tilt_deg:
                if self.carrying is None:
                    self._pick_parcel()
                self.dest = self.order
                self._takeoff()
        elif ph == "takeoff":
            if fm.state in ("hovering", "forward") and fm.ctrl is not None:
                fm._clearance = c.cruise_alt
                self._set("climb")
            elif fm.state == "walking" and t > 0.2:
                self._set("wait")  # the jump never reached its flight phase
        elif ph in ("climb", "cruise", "approach"):
            self._guide(dt, t)
        elif ph == "landing":
            if fm.state == "landing":
                self._hold_mat(dt)  # keep leaning into the wind while descending
            elif fm.state == "touchdown" and not self._touched:
                # feet down (adhesion on): drop the horizontal integrators, or they
                # drag the stuck fly sideways during the pitch-down and flip it
                self._touched = True
                if fm.ctrl is not None:
                    fm.ctrl.xy_int[:] = 0.0
                    fm.ctrl.target.pos = np.r_[self.sim.com()[:2], fm.ctrl.target.pos[2]]
            if fm.state == "walking":
                self._set("landed")
        elif ph == "landed":
            if t >= c.settle_s:
                self._judge_landing()
        elif ph == "drop":
            if t >= c.drop_s:
                self.dest = -1
                self._takeoff()
        elif ph in ("down", "wait"):
            self._maybe_retry(t)
        self._update_props()

    def _com_vel(self) -> np.ndarray:
        """COM velocity averaged over the last wingbeat (the controller's 5 kHz log)."""
        ctrl = self.fm.ctrl
        if ctrl is None or not ctrl.log:
            return self.sim.thorax_linvel()
        rows = list(ctrl.log)[-23:]
        return np.mean([r[4:7] for r in rows], axis=0)

    def _velocity_mode(self, on: bool) -> None:
        """Cruise: velocity control only (the position set point follows the fly, a
        stiffer velocity loop, like FlightMode's escape flight); approach: back to the
        position loop on the doormat."""
        fm = self.fm
        ctrl = fm.ctrl
        if ctrl is None:
            return
        ctrl.g = replace(fm._gains, xy_zeta=self.cfg.cruise_xy_zeta) if on else fm._gains

    def _guide(self, dt: float, t: float) -> None:
        sim, fm, c = self.sim, self.fm, self.cfg
        ctrl = fm.ctrl
        if ctrl is None or fm.state not in ("hovering", "forward"):
            return
        dest = self.landing_point(self.dest)
        com = sim.com()
        rel = dest - com[:2]
        d = float(np.hypot(*rel))
        top_max = max([h[2] for h in self.houses] + [c.depot_top])
        if self._t_leg0 is not None and sim.time - self._t_leg0 > c.leg_timeout_s:
            self._velocity_mode(False)
            self._land()
            return
        if self.phase == "climb":
            fm._clearance = c.cruise_alt
            if d > 1.0:
                fm._yaw_goal = math.atan2(rel[1], rel[0])
            err = abs(wrap_angle(fm._yaw_goal - ctrl.target.yaw))
            if com[2] > top_max + c.min_alt_margin and err < math.radians(c.turn_first_deg):
                self._velocity_mode(True)
                self._v = 0.0
                self._set("cruise")
            return
        if self.phase == "cruise":
            # velocity control along the bearing to the doormat, trapezoid profile
            if d > 1.0:
                fm._yaw_goal = math.atan2(rel[1], rel[0])
            err = abs(wrap_angle(fm._yaw_goal - ctrl.target.yaw))
            v_des = min(c.cruise_speed, math.sqrt(2.0 * c.decel * max(d - c.handover_mm, 0.0)))
            if err > math.radians(c.turn_first_deg):
                v_des = 0.0
            self._v = min(self._v + c.accel * dt, v_des)
            ctrl.target.pos = np.r_[com[:2], ctrl.target.pos[2]]
            ctrl.xy_int[:] = 0.0
            if d < c.descend_at_mm:
                fm._clearance = self.pad(self.dest)[2] + c.approach_clear
            fm._speed = self._v
            want = "forward" if self._v > 0 else "hovering"
            if fm.state != want:
                fm.state = want
            if d <= c.handover_mm + 0.2 or (d < 3 * c.handover_mm and self._v < 1.0 and t > 0.3):
                # hand over to the position loop on the doormat
                fm._speed = 0.0
                self._v = 0.0
                self._velocity_mode(False)
                self._approach_gains()
                self._lean = np.zeros(2)
                ctrl.target.pos = np.r_[dest, ctrl.target.pos[2]]
                ctrl.xy_int[:] = 0.0
                fm._clearance = self.pad(self.dest)[2] + c.approach_clear
                self._set("approach")
            return
        # approach: position loop on the doormat (stiffer gains + a slow set-point
        # integrator that leans the set point into the wind)
        fm._speed = 0.0
        if fm.state != "hovering":
            fm.state = "hovering"
        self._hold_mat(dt)
        fm._clearance = self.pad(self.dest)[2] + c.approach_clear
        vxy = float(np.hypot(*self._com_vel()[:2]))
        dz = abs(com[2] - fm._clearance)
        if (d < c.land_radius and vxy < c.land_max_speed and dz < 0.6) or t > c.approach_timeout_s:
            self._land()

    def _hold_mat(self, dt: float) -> None:
        """Position set point on the doormat + the wind-lean integrator."""
        c, fm = self.cfg, self.fm
        ctrl = fm.ctrl
        if ctrl is None:
            return
        dest = self.landing_point(self.dest)
        err = dest - self.sim.com()[:2]
        if float(np.hypot(*err)) < c.lean_zone_mm:  # anti-windup: only close to the mat
            self._lean = np.clip(self._lean + c.lean_ki * err * dt, -c.lean_max, c.lean_max)
        ctrl.target.pos = np.r_[dest + self._lean, ctrl.target.pos[2]]

    def _approach_gains(self) -> None:
        fm, c = self.fm, self.cfg
        g = replace(fm._gains, xy_wn=c.approach_xy_wn, xy_zeta=c.approach_xy_zeta,
                    xy_ki=c.approach_xy_ki)
        fm._gains = g  # (FlightMode's landing keeps them)
        if fm.ctrl is not None:
            fm.ctrl.g = g

    def _land(self) -> None:
        msg = self.fm.land(source="job")
        self._touched = False
        if msg == "[flight] landing":
            self._set("landing")

    def _judge_landing(self) -> None:
        if self.fm.state != "walking":
            return
        if self.on_pad(self.dest) and self.sim.tilt_deg() < 45.0 and not self.fly_down():
            if self.dest >= 0 and self.carrying is not None:
                self._deliver()
                self._set("drop")
            elif self.dest >= 0:
                self._set("drop")  # (after a reset mid-route: nothing to drop)
            else:
                self._begin_load()
            return
        if self.sim.tilt_deg() > 60.0:  # flipped over on touchdown
            self.n_crash += 1
            self._flash("CRASH LANDING!", f"at {self.dest_name()}", (255, 110, 90))
            self.session.log_event("crash_landing", job=self.name, target=self.dest_name())
            self._set("down")
            return
        self.n_missed += 1
        where = next((self.dest_name(i) + " roof" for i in range(-1, len(self.houses))
                      if self.on_pad(i)), "the street")
        self._flash("MISSED LANDING", f"on {where} - retry", (255, 200, 90))
        self.session.log_event("missed_landing", job=self.name, target=self.dest_name(),
                               where=where)
        self._set("wait")

    def _maybe_retry(self, t: float) -> None:
        sim, fm, c = self.sim, self.fm, self.cfg
        if fm.state != "walking" or self.fly_down():
            self._upright_since = None
            return
        if sim.tilt_deg() > c.max_retry_tilt_deg:
            self._upright_since = None
            return
        if self._upright_since is None:
            self._upright_since = sim.time
        if sim.time - self._upright_since >= c.retry_after_s and t >= c.retry_after_s:
            self._upright_since = None
            if self.on_pad(self.dest):  # (e.g. a crash that ended on the right roof)
                self._set("landed")
                self.t_phase = sim.time - c.settle_s
                return
            self._takeoff()

    # ------------------------------------------------------------ props / wind
    def _update_wind(self) -> None:
        c = self.cfg
        if c.wind_max <= 0:
            return
        s = self.run_time() * 2 * math.pi / c.wind_period_s
        mag = c.wind_max * (0.55 + 0.45 * math.sin(0.61 * s + 1.0))
        ang = 0.9 + 0.8 * math.sin(0.37 * s) + 0.3 * math.sin(1.3 * s)
        self._wind[:] = (mag * math.cos(ang), mag * math.sin(ang))
        self.sim.model.opt.wind[:2] = self._wind

    def _place_parcel(self, k: int, pos) -> None:
        self.sim.data.mocap_pos[self.mocap_parcel[k]] = pos

    def _update_props(self) -> None:
        sim, c = self.sim, self.cfg
        d, m = sim.data, sim.model
        now = sim.time
        for k, st in enumerate(self.parcel_state):
            mid = self.mocap_parcel[k]
            anim = self.parcel_anim[k]
            if st == "carried":
                com = sim.com()
                zh = c.parcel_half * 0.8
                # hanging on its string in the air; set down under the fly on a roof
                floor = self.ground_height(float(com[0]), float(com[1])) + zh + 0.005
                goal = np.array([com[0], com[1], max(com[2] + c.carry_dz, floor)])
                if anim is not None:  # sliding from the shelf to the fly
                    a = min(1.0, (now - anim[0]) / max(c.load_s * 0.8, 1e-3))
                    a = a * a * (3 - 2 * a)
                    p = (1 - a) * anim[1] + a * goal
                    p[2] += 1.2 * math.sin(math.pi * a)
                    d.mocap_pos[mid] = p
                    if a >= 1.0:
                        self.parcel_anim[k] = None
                else:
                    d.mocap_pos[mid] = goal
                yaw = self.fm.ctrl.target.yaw if self.fm.ctrl is not None else sim.heading()
                d.mocap_quat[mid] = quat_axis_angle((0, 0, 1), yaw)
                L = float(com[2] - 0.2 - (d.mocap_pos[mid][2] + zh))
                g = self.string_gid[k]
                if anim is None and L > 0.05:
                    m.geom_rgba[g, 3] = 1.0
                    m.geom_size[g, 1] = L / 2
                    m.geom_pos[g, 2] = zh + L / 2
                else:
                    m.geom_rgba[g, 3] = 0.0
            else:
                m.geom_rgba[self.string_gid[k], 3] = 0.0
                if anim is not None and len(anim) == 3:
                    a = min(1.0, (now - anim[0]) / max(c.drop_s * 0.6, 1e-3))
                    a = a * a * (3 - 2 * a)
                    d.mocap_pos[mid] = (1 - a) * anim[1] + a * anim[2]
                    if a >= 1.0:
                        self.parcel_anim[k] = None
        rt = self.run_time()
        for i, gid in enumerate(self.lamp_gid):
            m.geom_matid[gid] = self.mat_on if rt < self._lamp_until[i] else self.mat_off
            mq = self.mocap_flag[i]
            up = self._flag_up[i] and self.parcel_state.count(("mat", i)) > 0
            d.mocap_quat[mq] = (quat_axis_angle_mul(self.flag_quat0[i], (0, 1, 0), -1.4)
                                if up else self.flag_quat0[i])
        w = self._wind
        yaw = math.atan2(w[1], w[0])
        droop = 1.2 * (1.0 - min(1.0, float(np.hypot(*w)) / 30.0))
        d.mocap_quat[self.mocap_sock] = _qmul(quat_axis_angle((0, 0, 1), yaw),
                                              quat_axis_angle((0, 1, 0), droop))

    # ------------------------------------------------------------ camera / HUD
    def camera_target(self) -> np.ndarray:
        p = self.sim.com()
        return p + np.array([0.0, 0.0, 1.5])

    def camera_preset(self) -> CameraPreset:
        c = self.cfg
        fm = self.fm
        now = self.sim.time
        if fm.ctrl is not None and fm.airborne:
            goal = fm.ctrl.target.yaw
        else:
            dest = self.landing_point(self.order if self.dest < 0 and self.phase == "load"
                                      else self.dest)
            rel = dest - self.sim.com()[:2]
            goal = math.atan2(rel[1], rel[0]) if np.hypot(*rel) > 3.0 else self._cam_yaw
        if self._cam_t is None or now < self._cam_t:
            self._cam_yaw = goal
        else:
            a = 1.0 - math.exp(-(now - self._cam_t) / max(c.cam_yaw_tau_s, 1e-3))
            self._cam_yaw += a * wrap_angle(goal - self._cam_yaw)
        self._cam_t = now
        return CameraPreset(azimuth=math.degrees(self._cam_yaw), elevation=c.cam_elevation,
                            distance=c.cam_distance, tau_s=0.25)

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """The DELIVERED / LOADED / MISSED caption (a screen overlay, not the scene)."""
        if t >= self._caption_until or not self.caption:
            return frame
        img = np.ascontiguousarray(frame)
        h, w = img.shape[:2]
        s = w / 720.0
        font = cv2.FONT_HERSHEY_DUPLEX
        for text, scale, y in ((self.caption, 1.3 * s, int(h * 0.16)),
                               (self.caption_sub, 0.62 * s, int(h * 0.16 + 36 * s))):
            if not text:
                continue
            th = max(1, int(round(2.2 * s)))
            (tw, _), _ = cv2.getTextSize(text, font, scale, th)
            org = ((w - tw) // 2, y)
            cv2.putText(img, text, org, font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, text, org, font, scale, self._caption_rgb, th, cv2.LINE_AA)
        return img

    def on_time_pct(self) -> float:
        return 100.0 * self.n_on_time / self.n_delivered if self.n_delivered else 100.0

    def job_hud_lines(self) -> list[str]:
        fm = self.fm
        dest = self.landing_point(self.dest)
        dist = float(np.hypot(*(dest - self.sim.com()[:2])))
        return [
            f"route: {self.dest_name()} ({dist:4.0f} mm)   parcel for "
            f"{self.dest_name(self.order) if self.carrying is not None else '-'}   [{self.phase}]",
            f"flight time {self.flight_time:6.1f} s   distance flown {self.distance / 1000:6.2f} m"
            f"   on time {self.on_time_pct():5.1f} %",
            f"missed landings {self.n_missed}   crash landings {self.n_crash}   "
            f"take-offs {self.n_takeoffs}   wind {np.hypot(*self._wind):3.0f} mm/s",
            fm.hud_line(),
            "real flapping-wing flight; parcel carried kinematically (no mass)",
        ]

    def job_stats(self) -> dict[str, Any]:
        return {
            "delivered": self.n_delivered, "on_time_pct": round(self.on_time_pct(), 1),
            "flight_time_s": round(self.flight_time, 2),
            "distance_mm": round(self.distance, 1),
            "missed_landings": self.n_missed, "crash_landings": self.n_crash,
            "takeoffs": self.n_takeoffs, "landings": self.n_landings,
            "flight_aborts": self.n_flight_aborts,
            "per_house": list(self.per_house),
            "best_delivery_s": (round(self.best_leg_s, 2) if math.isfinite(self.best_leg_s)
                                else None),
        }


def _qmul(q1, q2):
    out = np.empty(4)
    mj.mju_mulQuat(out, np.asarray(q1, float), np.asarray(q2, float))
    return out


def quat_axis_angle_mul(q0, axis, angle):
    return _qmul(q0, quat_axis_angle(axis, angle))
