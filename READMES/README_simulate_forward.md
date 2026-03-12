# Forward Flow-Dye Simulator Code Guide (`simulate_forward.py`) 

## 1. Purpose

`simulate_forward.py` implements a complete synthetic forward pipeline:

`Evolution Model -> Optical Model -> Camera Model -> Export`

It evolves a latent physical state (particles + dye), renders intensity on an `x-z` image plane, converts that intensity into `uint8` camera-like frames, and saves a GIF.

The frame origin is centered in the `x-z` plane.

## 2. Core Data Structures

### 2.1 `SimConfig`

`SimConfig` is the single parameter container. Main groups:

- Time / size: `T`, `dt`, `H`, `W`, `N`
- Physical domain: `Lx`, `zmin`, `zmax`
- Velocity field: `A`, `k`, `gamma`
- Stochastic transport: `particle_noise_sigma`, `dye_kappa`
- Out-of-plane and sheet gating:
  - `enable_out_of_plane` (backward-compatibility switch)
  - `enable_sheet_gating`
  - `sheet_center_y`, `sheet_thickness`
  - `y_noise_sigma`, `y_kill`
- Optical rendering: `psf_sigma_px`, `particle_amp`, `dye_beta`, `dye_alpha`, `light_source_x_frac`, `light_source_z_above_frac`, `dye_blur_sigma_px`
- Camera / exposure: `use_camera_model`, `bg`, `gain`, `read_sigma`, `auto_exposure`, `exposure_percentile`
- Reproducibility: `seed`

### 2.2 `State`

`State` stores the evolving latent variables:

- `xp`: particle x-coordinate array `(N,)`
- `zp`: particle z-coordinate array `(N,)`
- `y`: particle depth coordinate `(N,)`
- `c`: dye concentration field `(H, W)`

## 3. Coordinate and Boundary Conventions

- `x`: periodic centered interval `[-Lx/2, Lx/2)`
- `z`: bounded interval `[zmin, zmax]`
- `y`: depth/camera axis, used for visibility gating

Helper `wrap_x_centered` enforces periodic `x` and is reused by particle and dye transport.

## 4. Evolution Model (Section-by-Section)

### 4.1 Velocity field: `vel_u_w`

`vel_u_w(x, z, t, A, k, gamma)` returns velocity `(u, w)` on the `x-z` plane. The field is built from an analytic decaying mode:

- decay with height: `exp(-k|z|)`
- oscillation in `x`: `exp(i k x)`
- optional growth/decay in time: `exp(gamma t)`

This is the shared flow model for both particle advection and dye advection.

### 4.2 Particle advection: `advect_particles_rk2`

Particles are advanced by RK2 (midpoint method):

1. evaluate velocity at current position
2. predict midpoint
3. evaluate midpoint velocity
4. advance full step

Then:

- add Gaussian process noise if `particle_noise_sigma > 0`
- wrap `x` periodically
- clip `z` to `[zmin, zmax]`

### 4.3 Dye advection: `advect_dye_semilag`

Dye uses semi-Lagrangian backtracing:

1. build grid `(X, Z)`
2. evaluate velocity `(u, w)` on grid
3. backtrace to `(Xb, Zb)`
4. interpolate previous `c` at backtraced points via `bilinear_sample`

`bilinear_sample` supports periodic indexing in `x` and clamped indexing in `z`.

If `dye_kappa > 0`, an explicit diffusion term is added via a finite-difference Laplacian:

- periodic neighbors in `x` (`np.roll`)
- edge-stable handling in `z` (`np.pad(..., mode="edge")`)
- clamp to nonnegative concentration

### 4.4 Out-of-plane dynamics and visibility

- `update_y_depth`: random-walk depth update
  - `y += Normal(0, y_noise_sigma * sqrt(dt))`
- `respawn`: re-initializes particles whose depth exceeds kill threshold
  - kill condition in caller: `abs(y - sheet_center_y) > y_kill`
- `visible_mask_y`: visibility test
  - if `enable_sheet_gating=False`: all visible
  - otherwise visible iff `abs(y - sheet_center_y) <= sheet_thickness/2`

Compatibility helpers:

- `update_out_of_plane` is an alias wrapper around `update_y_depth`
- `visible_mask` is an alias wrapper around `visible_mask_y`

### 4.5 One-step evolution: `step_evolution`

Per step order is:

1. update particles (`advect_particles_rk2`)
2. update dye (`advect_dye_semilag`)
3. if `enable_out_of_plane=True`, update `y` and respawn killed particles
4. compute visibility mask with `visible_mask_y`

Returns updated `(state, vis)`.

## 5. Optical Model

### 5.1 `gaussian_blur_fft`

Applies Gaussian blur in Fourier domain:

- FFT image
- multiply by Gaussian kernel in frequency space
- inverse FFT

### 5.2 `render_particles`

- map physical particle coordinates to pixel indices
- accumulate intensity impulses with `np.add.at`
- blur by `psf_sigma_px`

Only currently visible particles are rendered (via `state.xp[vis]`, `state.zp[vis]`).

### 5.3 `render_dye`

Builds dye intensity from field `c` using a point-like light source model:

- source position from `light_source_x_frac` and `light_source_z_above_frac`
- distance-based attenuation `L = 1/(d^2 + eps)`
- intensity term:
  - `I = dye_beta * c * L * exp(-dye_alpha * d * c)`
- optional blur `dye_blur_sigma_px`

### 5.4 `render_total_intensity`

Combines optical contributions:

- `I_total = I_particles + I_dye`

## 6. Camera Model and Encoding

### 6.1 `camera_model`

A simple noisy sensor model:

- add background (`bg`)
- Poisson shot noise with `gain`
- Gaussian read noise (`read_sigma`)
- clip to `[0, 255]`, cast to `uint8`

### 6.2 `auto_exposure_to_uint8`

Percentile normalization path:

- compute high percentile (`exposure_percentile`)
- scale intensity to 8-bit range

### 6.3 `encode_frame`

Decision logic:

- if `use_camera_model=True` and `auto_exposure=False` -> `camera_model`
- otherwise -> `auto_exposure_to_uint8`

So by default (`auto_exposure=True`), auto-exposure path is used.

## 7. Initialization and Diagnostics

### 7.1 `init_state`

Initializes deterministic random state from `seed`, then:

- `xp`: uniform on centered interval `[-Lx/2, Lx/2)`
- `zp`: Gaussian around 0, clipped to `[zmin, zmax]`
- `y`: uniform around `sheet_center_y ± 0.25*sheet_thickness`
- grid:
  - `xs = linspace(-Lx/2, Lx/2, W, endpoint=False)`
  - `zs = linspace(zmin, zmax, H)`
- dye field `c`: Gaussian blob `exp(-(X^2 + (Z-0.15)^2)/0.02)`

### 7.2 Diagnostics

`init_diagnostics` creates lists for:

- `I_min`, `I_max`, `I_mean`, `visible_frac`

`append_diagnostics` appends per-frame summary values.

## 8. Main Simulation Loop

`forward_simulator(cfg)` performs:

1. initialize `state, xs, zs`
2. allocate `video` array `(T, H, W)` as `uint8`
3. for each time step:
   - `state, vis = step_evolution(...)`
   - `I = render_total_intensity(...)`
   - `frame = encode_frame(I, cfg)`
   - write frame to `video`
   - append diagnostics
4. return `(video, state, diag)`

## 9. Export and Script Entry Point

### 9.1 `export_video`

Current export function writes only a GIF:

- output path: `<out_dir>/<base>.gif`
- encoder: `imageio.mimsave`

### 9.2 `main`

`main()` currently uses default config only (no argument parsing in this version):

- runs `forward_simulator(SimConfig())`
- saves to `<script_dir>/Outputs_simulate`
- prints video shape, average visible fraction, and save path

## 10. Call Graph

```text
main
  -> forward_simulator
       -> init_state
       -> init_diagnostics
       -> loop over T:
            -> step_evolution
                 -> advect_particles_rk2
                      -> vel_u_w
                 -> advect_dye_semilag
                      -> vel_u_w
                      -> bilinear_sample
                 -> update_y_depth (if enable_out_of_plane)
                 -> respawn (if enable_out_of_plane)
                 -> visible_mask_y
            -> render_total_intensity
                 -> render_particles
                      -> gaussian_blur_fft
                 -> render_dye
                      -> gaussian_blur_fft
            -> encode_frame
                 -> camera_model or auto_exposure_to_uint8
            -> append_diagnostics
       -> return video, state, diag
  -> export_video
```

## 11. Known Behavior Notes

- `c` is actively evolved every step in this version (semi-Lagrangian + optional diffusion).
- `enable_out_of_plane=False` disables `y` random walk and respawn, but visibility still follows `visible_mask_y` based on current `y` and gating settings.
- No MP4 export path exists in this file version; export is GIF-only.
