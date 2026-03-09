# GIF Key Frame Extractor (`extract_key_frames.py`) ($Hongze$ $Lin$)

## 1. Purpose

`extract_key_frames.py` extracts 3 key frames from each of two output GIF files:

1. Frame 0 (first frame)
2. Middle frame
3. Last frame

The script automatically creates the target folders and saves the extracted images.

---

## 2. Default Input Files

The script reads these two GIF files by default:

- `Outputs_simulate/sim.gif`
- `Outputs_pipeline/sim_pipeline.gif`

---

## 3. Output Directory Structure

After running, the script generates:

```text
Key_Outputs_image/
├── Key_simulate/
│   ├── sim_frame0.png
│   ├── sim_frame_mid.png
│   └── sim_frame_last.png
└── Key_pipeline/
    ├── sim_pipeline_frame0.png
    ├── sim_pipeline_frame_mid.png
    └── sim_pipeline_frame_last.png
```

---

## 4. How to Run

Run from the project root directory:

```bash
python3 extract_key_frames.py
```

---

## 5. Dependency

This script requires `Pillow` (`PIL`):

```bash
pip install pillow
```

---

## 6. Logic Summary

For each GIF:

1. Read total frame count `n_frames`
2. Compute key frame indices:
   - first frame: `0`
   - middle frame: `n_frames // 2`
   - last frame: `n_frames - 1`
3. Use `seek` to load each frame and save it as PNG

---

## 7. Notes

- If an input GIF path does not exist, the script raises `FileNotFoundError`.
- Existing output files with the same names will be overwritten.
- The middle frame uses integer floor division: `n_frames // 2`.
