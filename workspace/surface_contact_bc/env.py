"""A point mass, an impedance controller, and a compliant line contact.

The policy commands an absolute target ``q``; ``x`` is the physical point.
``n @ x - c >= 0`` is free space. Negative distances mean penetration.
All methods use the same low-level controller and receive the same observation.
"""

from dataclasses import asdict, dataclass, field

import numpy as np


OBS_FIELDS = (
    "x", "y", "vx", "vy", "normal_x", "normal_y", "surface_offset",
    "signed_distance", "tangential_goal", "normal_force", "time",
)
OBS_DIM = len(OBS_FIELDS)
APPROACH_END = 1.2
ENGAGE_END = 1.8
SLIDE_END = 4.8


@dataclass
class Physics:
    mass: float = 1.0
    kp: float = 120.0
    kd: float = 20.0
    contact_stiffness: float = 500.0
    contact_damping: float = 15.0
    friction: float = 0.2
    dt: float = 0.002
    control_dt: float = 0.04


@dataclass
class Context:
    normal: np.ndarray
    offset: float
    goal: float
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    physics: Physics = field(default_factory=Physics)
    duration: float = 6.0
    seed: int = 0

    @property
    def tangent(self):
        return np.array([self.normal[1], -self.normal[0]])

    def to_dict(self):
        value = asdict(self)
        for name in ("normal", "initial_position", "initial_velocity"):
            value[name] = np.asarray(value[name]).tolist()
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name in ("normal", "initial_position", "initial_velocity"):
            value[name] = np.asarray(value[name], dtype=np.float64)
        value["physics"] = Physics(**value["physics"])
        return cls(**value)


def sample_context(seed, suite="id", value=None):
    """Sample hidden dynamics and observed geometry; OOD values never train.

    Training angles are {-20, 0, 20} degrees from the upward normal.
    ``unseen_orientation`` uses ±45 degrees by default, or ``value`` degrees.
    Individual physics suites use ``value`` as a nominal-parameter multiplier.
    The intended shift range is 0.5–2 for mass/gains/damping, 0.5–3 for
    stiffness, and 0–3 for friction. Integration is tested over these ranges.
    """
    rng = np.random.default_rng(seed)
    angle = float(rng.choice([-20.0, 0.0, 20.0]))
    if suite in ("unseen_orientation", "orientation"):
        angle = float(value) if value is not None else float(rng.choice([-45., 45.]))
        if any(np.isclose(angle, train_angle) for train_angle in (-20, 0, 20)):
            raise ValueError("An unseen orientation must differ from training angles.")
    radians = np.deg2rad(angle)
    normal = np.array([np.sin(radians), np.cos(radians)])
    tangent = np.array([normal[1], -normal[0]])
    physics = Physics(
        mass=rng.uniform(0.9, 1.1), kp=rng.uniform(108., 132.),
        kd=rng.uniform(18., 22.), contact_stiffness=rng.uniform(450., 550.),
        contact_damping=rng.uniform(12., 18.), friction=rng.uniform(.15, .25),
    )
    nominal = Physics()
    fields = {"mass": "mass", "stiffness": "contact_stiffness",
              "damping": "contact_damping", "friction": "friction",
              "contact_stiffness": "contact_stiffness",
              "contact_damping": "contact_damping"}
    if suite in fields:
        name = fields[suite]
        setattr(physics, name, getattr(nominal, name) * (2. if value is None else value))
    elif suite in ("gains", "controller_gains"):
        multiplier = 1.7 if value is None else float(value)
        physics.kp, physics.kd = nominal.kp * multiplier, nominal.kd * multiplier
    elif suite in ("contact_physics_shift", "physics"):
        multiplier = 2. if value is None else float(value)
        physics.mass *= multiplier
        physics.contact_stiffness *= multiplier
        physics.contact_damping *= multiplier
        physics.friction *= multiplier
        physics.kp /= np.sqrt(multiplier)
        physics.kd /= np.sqrt(multiplier)
    elif suite not in ("id", "train", "val", "test", "unseen_orientation",
                       "orientation", "normal_impulse_sweep", "impulse"):
        raise ValueError(f"Unknown evaluation suite: {suite}")
    start_tangent = rng.uniform(-.3, .3)
    goal = start_tangent + rng.choice([-1., 1.]) * rng.uniform(.3, .55)
    offset = rng.uniform(-.08, .08)
    initial_position = (offset + rng.uniform(.16, .25)) * normal + start_tangent * tangent
    return Context(normal, offset, float(goal), initial_position,
                   rng.uniform(-.015, .015, size=2), physics, seed=int(seed))


def phase(time):
    return ("approach" if time < APPROACH_END else "engage" if time < ENGAGE_END
            else "slide" if time < SLIDE_END else "stop")


class SurfaceContactEnv:
    """Deterministic semi-implicit Euler simulation, with held target actions."""

    def __init__(self, context):
        self.context = context
        physics = context.physics
        if min(physics.mass, physics.kp, physics.dt, physics.control_dt) <= 0:
            raise ValueError("Mass, controller stiffness, and timesteps must be positive.")
        if min(physics.kd, physics.contact_stiffness, physics.contact_damping,
               physics.friction) < 0:
            raise ValueError("Damping, contact stiffness, and friction must be nonnegative.")
        if not np.isclose(np.linalg.norm(context.normal), 1.):
            raise ValueError("The surface normal must be a unit vector.")
        self.reset()

    def reset(self):
        self.x = self.context.initial_position.copy()
        self.v = self.context.initial_velocity.copy()
        self.time, self.steps = 0., 0
        self.last_info = self.contact_info()
        return self.observe()

    def contact_info(self):
        context, physics = self.context, self.context.physics
        distance = float(context.normal @ self.x - context.offset)
        normal_velocity = float(context.normal @ self.v)
        tangent_velocity = float(context.tangent @ self.v)
        penetration = max(-distance, 0.)
        # No attraction: damping cannot pull the point back into the surface.
        force = (max(physics.contact_stiffness * penetration
                     - physics.contact_damping * normal_velocity, 0.)
                 if penetration > 0. else 0.)
        friction = -physics.friction * force * np.tanh(tangent_velocity / .02)
        return dict(normal_force=force, penetration=penetration, contact=penetration > 0.,
                    friction_force=float(friction), signed_distance=distance,
                    normal_velocity=normal_velocity,
                    tangential_position=float(context.tangent @ self.x),
                    tangential_velocity=tangent_velocity, phase=phase(self.time),
                    peak_normal_force=force, peak_penetration=penetration)

    def observe(self):
        info = self.contact_info()
        return np.asarray([*self.x, *self.v, *self.context.normal, self.context.offset,
                           info["signed_distance"], self.context.goal,
                           info["normal_force"], self.time], dtype=np.float32)

    def step(self, target, impulse=0.):
        """Advance one control interval; impulse is momentum along ``normal``."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("Expected one finite absolute Cartesian target of shape (2,).")
        context, physics = self.context, self.context.physics
        self.v += context.normal * float(impulse) / physics.mass
        substeps = int(np.ceil(physics.control_dt / physics.dt))
        dt = physics.control_dt / substeps
        peak, peak_penetration = 0., 0.
        for _ in range(substeps):
            info = self.contact_info()
            force = (physics.kp * (target - self.x) - physics.kd * self.v
                     + info["normal_force"] * context.normal
                     + info["friction_force"] * context.tangent)
            self.v += dt * force / physics.mass
            self.x += dt * self.v
            peak = max(peak, info["normal_force"])
            peak_penetration = max(peak_penetration, info["penetration"])
        self.steps += 1
        self.time = self.steps * physics.control_dt
        self.last_info = self.contact_info()
        self.last_info["peak_normal_force"] = max(peak, self.last_info["normal_force"])
        self.last_info["peak_penetration"] = max(peak_penetration, self.last_info["penetration"])
        return self.observe()


def expert_target(env):
    """Independent phase-based physical impedance expert, not a reference path.

    A bounded desired physical velocity approaches/engages the surface and then
    slides toward the endpoint. Inverse impedance compensates contact/friction.
    Physics parameters are used by this expert, but never placed in observations.
    """
    context, physics = env.context, env.context.physics
    info = env.contact_info()
    engagement = np.clip((env.time - APPROACH_END) / (ENGAGE_END - APPROACH_END), 0., 1.)
    engagement = engagement ** 2 * (3. - 2. * engagement)
    desired_distance = (1. - engagement) * .008 - engagement * .009
    normal_speed = .22 * np.tanh(4. * (desired_distance - info["signed_distance"]) / .22)
    tangent_speed = (0. if env.time < ENGAGE_END else
                     .28 * np.tanh(3. * (context.goal - info["tangential_position"]) / .28))
    desired_velocity = normal_speed * context.normal + tangent_speed * context.tangent
    acceleration = 12. * (desired_velocity - env.v)
    contact_force = (info["normal_force"] * context.normal
                     + info["friction_force"] * context.tangent)
    return env.x + (physics.mass * acceleration + physics.kd * env.v - contact_force) / physics.kp


def rollout_expert(context):
    env = SurfaceContactEnv(context)
    observations, actions, infos = [env.observe()], [], [env.last_info.copy()]
    for _ in range(int(np.ceil(context.duration / context.physics.control_dt))):
        target = expert_target(env)
        actions.append(target)
        observations.append(env.step(target))
        infos.append(env.last_info.copy())
    return dict(context=context.to_dict(), observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                infos={key: np.asarray([info[key] for info in infos]) for key in infos[0]})
