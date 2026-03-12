"""
Evolution Model
- Hongze Lin
--------------------------------------------------------------
Basic Coordinate Assumptions
Camera optical axis: +y
Sensor / pixel plane: x–z
Frame origin: (x=0, z=0) at the CENTER of the frame.

Variable Statement (same notation as simulate_forward / pipeline_forward):
  xp: particle x positions (width; out-of-paper direction), centered: [-Lx/2, Lx/2)
  zp: particle z positions (height), typically centered if zmin=-zmax
  y : particle y positions (depth/camera axis; used for sheet gating)
  c : dye concentration slice on x–z grid, shape (H, W)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path


# =========================================================
# Config / Variable Statement
# =========================================================

@dataclass
class SimConfig:
    # time / size
    T: int = 80
    dt: float = 0.01
    H: int = 384   # pixels along z
    W: int = 384   # pixels along x
    N: int = 1000

    # physical domain
    # x: width (perpendicular to drawing plane), centered at 0
    # y: length (camera optical axis)
    # z: height, typically centered at 0 if zmin=-zmax
    Lx: float = 1.0
    zmin: float = -0.5
    zmax: float = 0.5

    # placeholder velocity field on x–z slice
    A: float = 0.02
    k: float = 2 * np.pi / 1.0
    gamma: float = 0.0

    # particle dynamics noise (model error) for x–z advection
    particle_noise_sigma: float = 5e-4

    # dye dynamics on x–z light-sheet slice
    dye_kappa: float = 0.0  # diffusion
    # concentration-field interface parameters (from SimpleModel_evolution_solver)
    dye_interface_eps: float = 0.02
    dye_interface_delta: float = 0.05

    # light-sheet gating along y (camera axis)
    enable_sheet_gating: bool = True
    sheet_center_y: float = 0.0
    sheet_thickness: float = 0.02

    # y evolution (to create appear/disappear in the sheet)
    y_noise_sigma: float = 0.005
    y_kill: float = 0.06

    # optional minimal display mode: keep only one pixel value per frame
    minimize_to_single_pixel: bool = False
    single_pixel_x: int = -1  # -1 -> center pixel in x
    single_pixel_z: int = -1  # -1 -> center pixel in z

    seed: int = 1


@dataclass
class State:
    xp: np.ndarray  # (N,) x positions (centered)
    zp: np.ndarray  # (N,) z positions
    y: np.ndarray   # (N,) y positions (depth/camera axis)
    c: np.ndarray   # (H,W) dye slice c(x,z,t) on centered x grid


# =========================================================
# Coordinate centered periodic wrap in x
# =========================================================

def wrap_x_centered(x: np.ndarray, Lx: float) -> np.ndarray:
    """
    Wrap x into the centered periodic interval [-Lx/2, Lx/2).
    """
    return ((x + 0.5 * Lx) % Lx) - 0.5 * Lx


# =========================================================
# Velocity field on x–z slice 
# =========================================================

def vel_u_w(x, z, t, A, k, gamma):
    """
    Placeholder analytic field on x–z plane:
      u = dx/dt, w = dz/dt
    """
    decay = np.exp(-k * np.abs(z))
    phase = np.exp(1j * k * x)
    growth = np.exp(gamma * t)
    u = np.real(1j * k * A * decay * phase * growth)
    w = np.real(-k * A * np.sign(z) * decay * phase * growth)
    return u, w


# =========================================================
# Numerical advection and evolution steps
# =========================================================

def advect_particles_rk2(x, z, t, dt, cfg: SimConfig):
    """
    Particle advection on x–z slice using RK2, with optional process noise.
    - x: periodic in centered interval [-Lx/2, Lx/2)
    - z: clipped in [zmin, zmax]
    """
    u1, w1 = vel_u_w(x, z, t, cfg.A, cfg.k, cfg.gamma)
    xm = x + 0.5 * dt * u1
    zm = z + 0.5 * dt * w1

    u2, w2 = vel_u_w(xm, zm, t + 0.5 * dt, cfg.A, cfg.k, cfg.gamma)
    x_new = x + dt * u2
    z_new = z + dt * w2

    # model error / stochasticity
    if cfg.particle_noise_sigma > 0:
        x_new += np.random.normal(0.0, cfg.particle_noise_sigma, size=x_new.shape)
        z_new += np.random.normal(0.0, cfg.particle_noise_sigma, size=z_new.shape)

    # boundaries
    x_new = wrap_x_centered(x_new, cfg.Lx)
    z_new = np.clip(z_new, cfg.zmin, cfg.zmax)
    return x_new, z_new


def _concentration_rhs_upwind(c, xs, zs, t, cfg: SimConfig):
    """
    Upwind spatial discretisation of concentration transport:
        dC/dt = -u dC/dx - w dC/dz [+ kappa * Lap(C)].
    This follows the concentration-field treatment in
    SimpleModel_evolution_solver.ipynb.
    """
    X, Z = np.meshgrid(xs, zs)
    u, w = vel_u_w(X, Z, t, cfg.A, cfg.k, cfg.gamma)

    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    # Upwind in x (axis=1, centered x-grid).
    dcdx_b = np.zeros_like(c)
    dcdx_f = np.zeros_like(c)
    dcdx_b[:, 1:] = (c[:, 1:] - c[:, :-1]) / dx
    dcdx_f[:, :-1] = (c[:, 1:] - c[:, :-1]) / dx
    dcdx = np.where(u >= 0, dcdx_b, dcdx_f)

    # Upwind in z (axis=0).
    dcdz_b = np.zeros_like(c)
    dcdz_f = np.zeros_like(c)
    dcdz_b[1:, :] = (c[1:, :] - c[:-1, :]) / dz
    dcdz_f[:-1, :] = (c[1:, :] - c[:-1, :]) / dz
    dcdz = np.where(w >= 0, dcdz_b, dcdz_f)

    dcdt = -u * dcdx - w * dcdz

    if cfg.dye_kappa > 0:
        c_pad = np.pad(c, ((1, 1), (0, 0)), mode="edge")
        c_up = c_pad[0:-2, :]
        c_dn = c_pad[2:, :]
        c_lt = np.roll(c, 1, axis=1)
        c_rt = np.roll(c, -1, axis=1)
        lap = (c_lt - 2 * c + c_rt) / dx**2 + (c_up - 2 * c + c_dn) / dz**2
        dcdt = dcdt + cfg.dye_kappa * lap

    return dcdt


def advect_dye_semilag(c, xs, zs, t, dt, cfg: SimConfig):
    """
    One-step RK4 update for concentration field using upwind RHS.
    Function name is kept for backward compatibility.
    """
    k1 = _concentration_rhs_upwind(c, xs, zs, t, cfg)
    k2 = _concentration_rhs_upwind(c + 0.5 * dt * k1, xs, zs, t + 0.5 * dt, cfg)
    k3 = _concentration_rhs_upwind(c + 0.5 * dt * k2, xs, zs, t + 0.5 * dt, cfg)
    k4 = _concentration_rhs_upwind(c + dt * k3, xs, zs, t + dt, cfg)

    c_new = c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.clip(c_new, 0.0, 1.0).astype(np.float32)


def update_y_depth(y, dt, cfg: SimConfig):
    """
    Stochastic evolution of particle depth y (camera axis).
    """
    return y + np.random.normal(0.0, cfg.y_noise_sigma * np.sqrt(dt), size=y.shape)


def visible_mask_y(y, cfg: SimConfig):
    """
    Light-sheet gating along y:
      visible if |y - sheet_center_y| <= sheet_thickness/2
    """
    if not cfg.enable_sheet_gating:
        return np.ones_like(y, dtype=bool)
    return np.abs(y - cfg.sheet_center_y) <= 0.5 * cfg.sheet_thickness


def respawn(mask, state: State, cfg: SimConfig):
    """
    Respawn particles drifting too far in y.
    x is respawned uniformly on centered interval [-Lx/2, Lx/2).
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return state

    state.xp[idx] = np.random.uniform(-0.5 * cfg.Lx, 0.5 * cfg.Lx, size=idx.size).astype(np.float32)

    zp_new = np.random.normal(0.0, 0.12 * (cfg.zmax - cfg.zmin), size=idx.size)
    state.zp[idx] = np.clip(zp_new, cfg.zmin, cfg.zmax).astype(np.float32)

    state.y[idx] = np.random.uniform(
        cfg.sheet_center_y - 0.25 * cfg.sheet_thickness,
        cfg.sheet_center_y + 0.25 * cfg.sheet_thickness,
        size=idx.size,
    ).astype(np.float32)

    return state


# =========================================================
# Initialisation (centered coordinates)
# =========================================================

def init_state(cfg: SimConfig):
    """
    Initialise (x0) with frame-centered coordinates:
      xp in [-Lx/2, Lx/2)
      zp in [zmin, zmax] (often symmetric)
      y near sheet center
      c defined on centered x grid and z grid
    """
    np.random.seed(cfg.seed)

    xp = np.random.uniform(-0.5 * cfg.Lx, 0.5 * cfg.Lx, size=cfg.N).astype(np.float32)

    zp = np.random.normal(0.0, 0.12 * (cfg.zmax - cfg.zmin), size=cfg.N).astype(np.float32)
    zp = np.clip(zp, cfg.zmin, cfg.zmax)

    y = np.random.uniform(
        cfg.sheet_center_y - 0.25 * cfg.sheet_thickness,
        cfg.sheet_center_y + 0.25 * cfg.sheet_thickness,
        size=cfg.N,
    ).astype(np.float32)

    # centered x grid for the x–z slice
    xs = np.linspace(-0.5 * cfg.Lx, 0.5 * cfg.Lx, cfg.W, endpoint=False)
    zs = np.linspace(cfg.zmin, cfg.zmax, cfg.H)

    # Interface-like concentration field (from SimpleModel_evolution_solver).
    X, Z = np.meshgrid(xs, zs)
    eta = cfg.dye_interface_eps * np.cos(cfg.k * X)
    c = 0.5 * (1.0 + np.tanh((Z - eta) / cfg.dye_interface_delta))
    c = c.astype(np.float32)

    return State(xp=xp, zp=zp, y=y, c=c), xs, zs


# =========================================================
# One-step evolution: x_t -> x_{t+dt}
# =========================================================

def step_evolution(state: State, xs, zs, t, cfg: SimConfig):
    """
    Advance the latent state by one time step (Evolution model only).

    Order matches simulate_forward/pipeline_forward logic:
      1) advect particles in x–z (RK2 + noise)
      2) advect concentration c(x,z,t) in x–z (upwind + RK4 + optional diffusion)
      3) update y-depth stochastic process
      4) respawn particles beyond y_kill
      5) compute visibility mask vis

    Returns:
      state (updated)
      vis   (bool mask, shape (N,))
    """
    state.xp, state.zp = advect_particles_rk2(state.xp, state.zp, t, cfg.dt, cfg)
    state.c = advect_dye_semilag(state.c, xs, zs, t, cfg.dt, cfg)

    state.y = update_y_depth(state.y, cfg.dt, cfg)
    kill = np.abs(state.y - cfg.sheet_center_y) > cfg.y_kill
    state = respawn(kill, state, cfg)

    vis = visible_mask_y(state.y, cfg)
    return state, vis


# =========================================================
# Quick sanity-run without rendering
# =========================================================

def run_evolution_only(cfg: SimConfig):
    state, xs, zs = init_state(cfg)
    t = 0.0
    visible_frac = []

    for _ in range(cfg.T):
        state, vis = step_evolution(state, xs, zs, t, cfg)
        visible_frac.append(float(np.mean(vis)))
        t += cfg.dt

    return state, {"visible_frac": visible_frac}


# =========================================================
# Visualisation / Export (2D only, aligned with simulate_forward.py style)
# =========================================================

def _state_to_uint8_frame(state: State, vis, cfg: SimConfig):
    """
    Convert latent state to one grayscale uint8 frame.
    - base layer: concentration field c(x,z)
    - particle layer: visible particles splatted as bright impulses
    - per-frame percentile exposure mapping
    """
    img = state.c.astype(np.float32).copy()

    px = (state.xp[vis] + 0.5 * cfg.Lx) / cfg.Lx * (cfg.W - 1)
    pz = (state.zp[vis] - cfg.zmin) / (cfg.zmax - cfg.zmin) * (cfg.H - 1)
    ix = np.rint(px).astype(int)
    iz = np.rint(pz).astype(int)
    m = (ix >= 0) & (ix < cfg.W) & (iz >= 0) & (iz < cfg.H)
    np.add.at(img, (iz[m], ix[m]), 1.0)

    hi = np.percentile(img, 99.7)
    hi = max(hi, 1e-6)
    frame = np.clip(img / hi * 255.0, 0, 255).astype(np.uint8)
    return keep_single_pixel(frame, cfg)


def keep_single_pixel(frame, cfg):
    if not cfg.minimize_to_single_pixel:
        return frame

    ix = cfg.single_pixel_x if cfg.single_pixel_x >= 0 else (cfg.W // 2)
    iz = cfg.single_pixel_z if cfg.single_pixel_z >= 0 else (cfg.H // 2)
    ix = int(np.clip(ix, 0, cfg.W - 1))
    iz = int(np.clip(iz, 0, cfg.H - 1))

    out = np.zeros_like(frame)
    out[iz, ix] = frame[iz, ix]
    return out


def init_diagnostics():
    return dict(
        c_min=[],
        c_max=[],
        c_mean=[],
        visible_frac=[],
    )


def append_diagnostics(diag, c, vis):
    diag["c_min"].append(float(c.min()))
    diag["c_max"].append(float(c.max()))
    diag["c_mean"].append(float(c.mean()))
    diag["visible_frac"].append(float(np.mean(vis)))


def forward_evolution_simulator(cfg: SimConfig):
    state, xs, zs = init_state(cfg)

    video = np.zeros((cfg.T, cfg.H, cfg.W), dtype=np.uint8)
    diag = init_diagnostics()
    t = 0.0

    for n in range(cfg.T):
        state, vis = step_evolution(state, xs, zs, t, cfg)
        video[n] = _state_to_uint8_frame(state, vis, cfg)
        append_diagnostics(diag, state.c, vis)
        t += cfg.dt

    return video, state, diag


def export_video(video, out_dir, fps=20, base="evolution"):
    out_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    imageio.mimsave(
        str(out_dir / f"{base}.gif"),
        video,
        duration=1.0 / fps
    )


if __name__ == "__main__":
    cfg = SimConfig()

    video, state, diag = forward_evolution_simulator(cfg)

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "Outputs_evolution"
    export_video(video, out_dir)

    print("video shape:", video.shape)
    print("visible fraction:", np.mean(diag["visible_frac"]))
    print("saved to:", out_dir)
