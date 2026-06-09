# DXF Converter — Image / PDF → DXF for sign making

Command-line tools that turn a logo or letters (image **or** PDF) into a clean **DXF**
ready for laser/CNC cutting. The DXF separates **outer cut lines** from **inner holes**
(the counters inside letters like O, A, B, D), so a cutter knows what to cut out and
what to keep as a hole.

There are two scripts:

| Script | Use it for |
|---|---|
| **`img_2_dxf_high_res.py`** | **Recommended.** High-resolution, robust tracing — upscales the image, removes the background with K-Means, and produces clean single-line outlines even for thin strokes, gradients, and multi-colour logos. |
| `image_to_dxf.py` | Older/simpler version. |

## What the DXF contains
- **`OUTER_CUT`** layer (red) — the outer perimeter of every shape.
- **`INNER_CUT`** layer (blue) — the holes / counters inside letters.

Coordinates are written in **millimetres** (Y is flipped so the result is right-side-up
in CAD).

## Install
```
python -m venv .venv
.venv\Scripts\activate          # Windows  (or: source .venv/bin/activate)
pip install -r requirements.txt
```
Dependencies: OpenCV, ezdxf, Pillow, PyMuPDF (PDF support), NumPy.

## Usage
```
# Image, set the real-world width to 400 mm
python img_2_dxf_high_res.py logo.jpg --width-mm 400

# PDF (page 1) at 600 mm wide, and save a debug preview
python img_2_dxf_high_res.py logo.pdf --width-mm 600 --debug
```
The DXF is saved next to the input (same name, `.dxf`) unless you pass `-o`.

### Common options
| Option | Meaning |
|---|---|
| `--width-mm N` | Real output width in mm (sets the scale). |
| `-o FILE` | Output DXF path (default: input name + `.dxf`). |
| `--dpi N` | Render resolution for **PDF** input (default 300). |
| `--page N` | PDF page to use, 1-based (default 1). |
| `--epsilon N` | How much to simplify the outline, 0.1–5.0 (default 0.8). Lower = more detail. |
| `--smooth N` | Corner-rounding passes (Chaikin), default 3. |
| `--min-area N` | Ignore specks smaller than this (px², default 80). |
| `--threshold N` | Manual black/white cutoff 0–255 (default: automatic). |
| `--invert` / `--no-invert` | Force / disable colour inversion (auto-detects dark backgrounds). |
| `--no-upscale` | Skip the high-res upscale step. |
| `--gray` | Use the simple grayscale path instead of K-Means (for plain black-on-white art). |
| `--debug` | Save a contour preview PNG **and** the traced foreground mask, to inspect results. |

## How it works (pipeline)
1. **Load** the image, or render a PDF page to pixels (`--dpi`). Record mm-per-pixel.
2. **Upscale** to ~4000 px on the long side + sharpen, so thin strokes survive.
3. **Background removal** — K-Means groups the colours, marks near-white / border
   clusters as background, and merges everything else into **one** solid foreground mask
   (this avoids double-lines from anti-aliasing and gradients). `--gray` uses a simpler
   threshold path instead.
4. **Find & classify contours** — outlines are detected and sorted into **outer** shapes
   vs **inner** holes by nesting depth.
5. **Smooth** each outline (Douglas–Peucker simplify + Chaikin rounding).
6. **Write DXF** — outer lines on `OUTER_CUT`, holes on `INNER_CUT`, scaled to mm.

## Tips
- If the result is empty or wrong, run with `--debug` and open the `*_mask.png` to see
  exactly what was traced.
- For plain black text on white, `--gray` is often cleanest.
- Set `--width-mm` to your real sign width so the DXF comes out at the correct size.
