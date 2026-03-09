# Forward Goal-2 Pipeline Simulator (`pipeline_forward.py`) — Detailed README

## 1. Overview

`pipeline_forward.py` implements a forward simulator with an explicit imaging pipeline for Goal 2:

1. Illumination
2. Interaction
3. Attenuation
4. Imaging
5. Sensor

At each frame, it updates particles and dye, applies the pipeline above, generates an 8-bit frame, stores diagnostics, and (optionally) exports files.

This script is designed as a transparent, modular baseline for synthetic data generation and inverse-problem prototyping.

---

## 2. Pipeline Logic (Conceptual)

The simulator combines flow dynamics with optical modeling:

- **Dynamics (state evolution)**
  - Particle advection in a prescribed velocity field.
  - Dye field advection (semi-Lagrangian) with optional diffusion.
  - Optional out-of-plane motion and respawn.

- **Optical/sensor pipeline (Goal 2)**
  - **Illumination**: create a light field over the domain.
  - **Interaction**: convert particle/dye state to emitted/scattered signal.
  - **Attenuation**: apply Beer-Lambert-style transmission for dye signal.
  - **Imaging**: convert to image-like intensity using splatting + blur.
  - **Sensor**: convert ideal intensity to `uint8` (auto exposure and/or noise model).

---

## 3. Main Public Entry Points

### 3.1 `simulate_video_pipeline(...)`

Primary simulation API.

Returns:

- `video`: `np.ndarray`, shape `(T, H, W)`, `uint8`
- `final`: dict with final state arrays
  - `xp`, `zp`, `y`, `c`
- `diag`: diagnostic time series dictionary
  - `Imin`, `Imax`, `Imean`, `vis_frac`

### 3.2 `export_outputs(video, fps=20, base="sim")`

Exports frames/video to:

- `<script_dir>/Outputs_pipeline/<base>_frame0.png`
- `<script_dir>/Outputs_pipeline/<base>_frame_mid.png`
- `<script_dir>/Outputs_pipeline/<base>_frame_last.png`
- `<script_dir>/Outputs_pipeline/<base>.gif`
- `<script_dir>/Outputs_pipeline/<base>.mp4` (if MP4 backend available)

---

## 4. Module Structure

The code is organized into these sections.

### 4.1 Dynamics core

- `vel_u_w(...)`
  - Computes velocity field `(u, w)` using the analytic placeholder model.

- `advect_particles_rk2(...)`
  - RK2 (midpoint) particle integration.
  - Boundary handling: periodic in `x`, clipped in `z`.

- `bilinear_sample(...)`
  - Bilinear interpolation for semi-Lagrangian backtracing.

- `advect_dye_semilag(...)`
  - Semi-Lagrangian dye update.
  - Optional diffusion term via Laplacian.

### 4.2 Out-of-plane visibility model

Canonical names:

- `update_out_of_plane(...)`
- `visible_mask(...)`
- `respawn(...)`

Compatibility aliases (same behavior):

- `update_out_of_plane_y(...)`
- `visible_mask_from_sheet(...)`
- `respawn_particles(...)`

### 4.3 Imaging helpers

- `gaussian_blur_fft(...)`
- `render_particles(...)`

### 4.4 Goal-2 pipeline physics helpers

- `illumination_field(...)`
  - Supports:
    - `mode="point"`
    - `mode="gaussian_beam"`

- `dye_emission(...)`
- `beer_lambert_attenuation_path_integral(...)`

### 4.5 Sensor/exposure helpers

Canonical name:

- `auto_exposure_to_uint8(...)`

Compatibility alias:

- `auto_exposure_uint8(...)`

Also includes:

- `camera_model(...)`

### 4.6 Forward-loop internal modules

- `init_state(...)`
- `prepare_illumination(...)`
- `init_diagnostics(...)`
- `step_flow_dynamics(...)`
- `step_out_of_plane_and_visibility(...)`
- `compute_interaction_and_attenuation(...)`
- `render_total_intensity(...)`
- `encode_frame(...)`
- `append_diagnostics(...)`

These modules make the main simulation loop easier to read and align structurally with `simulate_forward.py`.

---

## 5. Detailed Per-Frame Execution Order

Inside `simulate_video_pipeline(...)`, each time step does:

1. **Flow dynamics update**
   - Update `xp, zp` with RK2.
   - Update dye field `c` via semi-Lagrangian advection (+ optional diffusion).

2. **Out-of-plane update and visibility**
   - Random-walk update for `y`.
   - Respawn particles that move beyond `y_kill`.
   - Build visibility mask `vis` for the sheet thickness.

3. **Pipeline Step 1: Illumination**
   - Use precomputed illumination field and particle illumination samples.

4. **Pipeline Step 2: Interaction**
   - Compute particle amplitudes from local illumination.
   - Compute dye emission from `c` and illumination grid.

5. **Pipeline Step 3: Attenuation**
   - Compute Beer-Lambert attenuation based on cumulative dye concentration.

6. **Pipeline Step 4: Imaging**
   - Render visible particles with splat + PSF blur.
   - Blur dye component.
   - Combine both channels into ideal intensity `I`.

7. **Pipeline Step 5: Sensor**
   - Apply auto exposure if enabled.
   - Apply camera noise model if enabled.
   - Store final `uint8` frame.

8. **Diagnostics**
   - Append `Imin`, `Imax`, `Imean`, `vis_frac`.

---

## 6. Important Parameters

The function signature of `simulate_video_pipeline(...)` exposes all major controls.

### 6.1 Timeline and grid

- `T`, `dt`
- `H`, `W`
- `N`

### 6.2 Physical domain

- `Lx`
- `zmin`, `zmax`

### 6.3 Velocity field

- `A`, `k`, `gamma`

### 6.4 Dynamics and model mismatch

- `particle_noise`
- `dye_kappa`

### 6.5 Out-of-plane process

- `enable_out_of_plane`
- `sheet_thickness`
- `y_noise_sigma`
- `y_kill`

### 6.6 Pipeline-specific optics

- Illumination geometry:
  - `illum_mode`
  - `light_x_frac`
  - `light_z_above_frac`
  - `beam_sigma`
  - `beam_depth_decay`

- Particle brightness:
  - `particle_base_amp`
  - `particle_illum_power`
  - `psf_sigma_px`

- Dye imaging:
  - `dye_beta`
  - `dye_alpha`
  - `dye_blur_sigma_px`

### 6.7 Sensor/display

- `use_camera_noise`
- `bg`, `gain`, `read_sigma`
- `auto_exposure`, `exposure_percentile`

### 6.8 Reproducibility

- `seed`

---

## 7. Diagnostics

`diag` contains a value per frame:

- `Imin`: minimum ideal intensity of the frame
- `Imax`: maximum ideal intensity of the frame
- `Imean`: mean ideal intensity of the frame
- `vis_frac`: fraction of particles visible in the sheet

These diagnostics are useful for sanity checks and comparing parameter settings.

---

## 8. Running the Script

This file currently runs from its `__main__` block with built-in parameter values.

Basic run:

```bash
python pipeline_forward.py
```

Expected console outputs include:

- video shape and dtype
- intensity summary over time
- average visible fraction
- export directory path

---

## 9. Programmatic Usage

Use as a module from another script:

```python
from pathlib import Path
import pipeline_forward as pf

video, final, diag = pf.simulate_video_pipeline(
    T=120,
    dt=0.02,
    H=256,
    W=256,
    N=1200,
    seed=42,
)

pf.export_outputs(video, fps=24, base="pipeline_caseA")
```

---

## 10. Output Behavior and Path

This script exports to a fixed path inside the script directory:

- `Outputs_pipeline`

That behavior is implemented in `export_outputs(...)` and is independent of your shell working directory.

---

## 11. Numerical/Model Notes

- Semi-Lagrangian advection is stable and simple, suitable for MVP-level synthetic generation.
- Particle rendering uses nearest-pixel splat + FFT Gaussian blur; this is efficient and produces realistic soft spots.
- Beer-Lambert attenuation is implemented as cumulative concentration integration along the assumed light path direction.
- The camera model combines Poisson shot noise and Gaussian read noise, then clips to `[0, 255]`.

---

## 12. Performance Notes

Most expensive operations are:

1. Full-frame FFT blurs.
2. Full-grid dye update and attenuation.
3. Particle splatting when `N` is large.

If runtime is high, first reduce `H/W`, then `T`, then `N`.

---

## 13. Common Tuning Scenarios

### Too dark

- Keep `auto_exposure=True`.
- Increase `particle_base_amp` or `dye_beta`.
- Lower `dye_alpha` if attenuation is too strong.

### Motion too weak

- Increase `A`.
- Increase `T` to observe longer evolution.

### Too much particle disappearance

- Increase `sheet_thickness`.
- Increase `y_kill`.
- Decrease `y_noise_sigma`.

### MP4 export warning

- Install `imageio-ffmpeg`.
- PNG and GIF are usually still available.

---

## 14. Relationship to `simulate_forward.py`

`pipeline_forward.py` and `simulate_forward.py` now share a more consistent modular structure and naming for overlapping concepts (dynamics/visibility/render/sensor/diagnostics).

`pipeline_forward.py` adds the explicit Goal-2 optical decomposition:

- illumination
- interaction
- attenuation
- imaging
- sensor

while preserving a similar flow of internal step functions.

---

## 15. Quick Checklist

When validating a run:

1. `video.dtype == uint8`
2. `video.shape == (T, H, W)`
3. `diag` lists all have length `T`
4. `Outputs_pipeline` contains expected files
5. `vis_frac` values are within `[0,1]`

---

## 16. Summary

`pipeline_forward.py` is a modular, explicit forward imaging simulator that couples:

- tractable flow dynamics,
- particle and dye representations,
- interpretable optical stages,
- and practical sensor modeling.

It is a good baseline for synthetic experiments where interpretability and controllability matter more than full CFD realism.
