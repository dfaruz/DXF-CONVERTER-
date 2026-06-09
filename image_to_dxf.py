#!/usr/bin/env python3
"""
image_to_dxf.py  —  Convert any image or PDF into a DXF cut file.

Output layers:
  OUTER_CUT (red)  — outer perimeters
  INNER_CUT (blue) — holes / counters (inside O, D, A, B, etc.)

Usage:
  python image_to_dxf.py input.jpg --width-mm 400
  python image_to_dxf.py logo.pdf  --width-mm 600 --dpi 300
  python image_to_dxf.py input.png --width-mm 400 --debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import ezdxf

# ── optional PDF support ───────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_input(path: str, dpi: int, page: int) -> tuple[np.ndarray, float]:
    """Return (bgr_uint8, mm_per_pixel).  mm_per_pixel is based on DPI; may be
    overridden later when --width-mm is known."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    ext = p.suffix.lower()

    if ext == ".pdf":
        if not HAS_FITZ:
            sys.exit("[ERROR] PDF support requires pymupdf. Run: pip install pymupdf")
        doc = fitz.open(str(p))
        idx = page - 1
        if idx >= len(doc):
            sys.exit(f"[ERROR] PDF has only {len(doc)} page(s); --page {page} is out of range.")
        pg  = doc[idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = pg.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            # Try via Pillow for exotic formats (CMYK JPEG, TIFF, WEBP…)
            try:
                from PIL import Image
                img = Image.open(str(p))
                if img.mode == "CMYK":
                    img = img.convert("RGB")
                elif img.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                else:
                    img = img.convert("RGB")
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                sys.exit(f"[ERROR] Cannot open image: {e}")

    mm_per_pixel = 25.4 / dpi
    return bgr, mm_per_pixel


# ══════════════════════════════════════════════════════════════════════════════
# 2.  AUTO-UPSCALE
# ══════════════════════════════════════════════════════════════════════════════

def _upscale_bgr(bgr: np.ndarray) -> np.ndarray:
    """If the image short side < 800 px, upscale to ~1200 px with Lanczos,
    then sharpen with an unsharp mask."""
    h, w = bgr.shape[:2]
    short = min(h, w)
    if short >= 800:
        return bgr

    scale = 1200 / short
    new_w, new_h = int(w * scale), int(h * scale)
    up = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # unsharp mask — sharpen edges
    blur    = cv2.GaussianBlur(up, (0, 0), 3)
    sharp   = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    print(f"[INFO] Upscaled {w}x{h} → {new_w}x{new_h}")
    return sharp


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SMART GRAY  (K-Means color separation for colorful logos)
# ══════════════════════════════════════════════════════════════════════════════

def _is_colorful(bgr: np.ndarray) -> bool:
    """Return True if the image looks like a multi-colour logo on a light background."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border_mask = np.zeros((h, w), dtype=np.uint8)
    bw = max(1, int(min(h, w) * 0.10))
    border_mask[:bw, :]  = 255
    border_mask[-bw:, :] = 255
    border_mask[:, :bw]  = 255
    border_mask[:, -bw:] = 255
    border_mean = float(gray[border_mask > 0].mean())

    return mean_sat > 15 and border_mean > 128


def _smart_gray(bgr: np.ndarray) -> list[np.ndarray]:
    """Return a list of binary masks — one per non-background colour cluster.
    Falls back to a single standard-grayscale mask for simple images."""
    if not _is_colorful(bgr):
        return []   # caller will use plain grayscale path

    img_h, img_w = bgr.shape[:2]

    # K-Means in Lab colour space
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    k        = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels   = labels.flatten().reshape(img_h, img_w)

    # Merge similar clusters (anti-aliasing fix)
    merge_threshold = 35.0
    cluster_map = list(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            ci = centers[cluster_map[i]]
            cj = centers[cluster_map[j]]
            dist = float(np.sqrt(sum((float(ci[d]) - float(cj[d])) ** 2 for d in range(3))))
            if dist < merge_threshold:
                target = cluster_map[i]
                for x in range(k):
                    if cluster_map[x] == cluster_map[j]:
                        cluster_map[x] = target

    unique_clusters = set(cluster_map)

    # Background = near-white (bright + low chroma)
    bg_clusters: set[int] = set()
    for idx in unique_clusters:
        L, a, b = centers[idx]
        chroma = np.sqrt((float(a) - 128) ** 2 + (float(b) - 128) ** 2)
        if L > 190 and chroma < 25:
            bg_clusters.add(idx)

    # Build one binary mask per foreground cluster
    masks: list[np.ndarray] = []
    for idx in unique_clusters:
        if idx in bg_clusters:
            continue
        members = {x for x in range(k) if cluster_map[x] == idx}
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for m in members:
            mask[labels == m] = 255

        # Remove tiny noise fragments
        n_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= 50:
                clean[cc_labels == lbl] = 255
        if clean.any():
            masks.append(clean)

    print(f"[INFO] K-Means colour separation: {len(masks)} foreground cluster(s)")
    return masks


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ORIENTATION FIX  (make sure background = black, logo = white)
# ══════════════════════════════════════════════════════════════════════════════

def _fix_binary_orientation(binary: np.ndarray) -> np.ndarray:
    """If corners of the binary image are bright (background wrongly = white),
    flip it so background = black."""
    h, w  = binary.shape
    px    = int(min(h, w) * 0.05) + 1
    corners = [
        binary[:px, :px],
        binary[:px, -px:],
        binary[-px:, :px],
        binary[-px:, -px:],
    ]
    mean_corner = float(np.concatenate([c.flatten() for c in corners]).mean())
    if mean_corner > 128:
        return cv2.bitwise_not(binary)
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PREPROCESS  (denoise → threshold → close gaps)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(gray: np.ndarray, threshold: int | None, force_invert: bool | None) -> np.ndarray:
    """Return a clean binary mask (logo = white, background = black)."""
    # Detect dark background via border sampling
    h, w  = gray.shape
    bw    = max(1, int(min(h, w) * 0.10))
    border = np.concatenate([
        gray[:bw, :].flatten(),
        gray[-bw:, :].flatten(),
        gray[:, :bw].flatten(),
        gray[:, -bw:].flatten(),
    ])
    dark_bg = border.mean() < 128

    if force_invert is True:
        gray = cv2.bitwise_not(gray)
    elif force_invert is None and dark_bg:
        print("[INFO] Dark background detected — auto-inverting")
        gray = cv2.bitwise_not(gray)

    # Denoise
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Threshold
    if threshold is not None:
        _, binary = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    else:
        # Try Otsu; fall back to adaptive if low-contrast
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio  = float(binary.mean()) / 255.0
        if fg_ratio > 0.85 or fg_ratio < 0.02:
            print("[INFO] Low-contrast image — using adaptive threshold")
            binary = cv2.adaptiveThreshold(
                filtered, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                blockSize=31, C=10
            )

    # Close small JPEG gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    binary = _fix_binary_orientation(binary)
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIND & CLASSIFY CONTOURS  (outer vs inner via depth)
# ══════════════════════════════════════════════════════════════════════════════

def find_and_classify(
    binary: np.ndarray,
    min_area: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (outer_contours, inner_contours).
    Even nesting depth → OUTER_CUT, odd depth → INNER_CUT."""
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    if hier is None:
        return [], []

    hier = hier[0]
    outer, inner = [], []

    def depth(idx: int) -> int:
        d = 0
        while hier[idx][3] != -1:
            idx = hier[idx][3]
            d  += 1
        return d

    for i, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) < min_area:
            continue
        d = depth(i)
        if d % 2 == 0:
            outer.append(cnt)
        else:
            inner.append(cnt)

    return outer, inner


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SMOOTH CONTOUR  (Douglas-Peucker + Chaikin corner-cutting)
# ══════════════════════════════════════════════════════════════════════════════

def smooth_contour(cnt: np.ndarray, epsilon: float, smooth: int = 3) -> np.ndarray:
    """Simplify then smooth a contour.
    epsilon  — Douglas-Peucker factor (per-mille of arc length; 1.0 is good default)
    smooth   — Chaikin iterations (0 = no smoothing, 3 = default)
    """
    arc       = cv2.arcLength(cnt, closed=True)
    approx_eps = (epsilon / 1000.0) * arc
    simplified = cv2.approxPolyDP(cnt, approx_eps, closed=True)
    pts        = simplified.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return pts

    for _ in range(smooth):
        n       = len(pts)
        new_pts = np.empty((n * 2, 2), dtype=np.float64)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts[i * 2]     = 0.75 * p0 + 0.25 * p1
            new_pts[i * 2 + 1] = 0.25 * p0 + 0.75 * p1
        pts = new_pts

    return pts


# ══════════════════════════════════════════════════════════════════════════════
# 8.  WRITE DXF
# ══════════════════════════════════════════════════════════════════════════════

def write_dxf(
    outer_list: list[list[np.ndarray]],
    inner_list: list[list[np.ndarray]],
    mm_per_pixel: float,
    img_height: int,
    out_path: str,
    epsilon: float,
    smooth: int,
) -> None:
    doc = ezdxf.new("R2010")
    doc.units = 4  # mm
    msp = doc.modelspace()

    doc.layers.add("OUTER_CUT", color=1)  # red
    doc.layers.add("INNER_CUT", color=5)  # blue

    def add_contours(cnts_group: list[list[np.ndarray]], layer: str) -> int:
        count = 0
        for cnts in cnts_group:
            for cnt in cnts:
                pts_px = smooth_contour(cnt, epsilon, smooth)
                if len(pts_px) < 3:
                    continue
                pts_mm = [
                    (float(x) * mm_per_pixel, float(img_height - y) * mm_per_pixel)
                    for x, y in pts_px
                ]
                msp.add_lwpolyline(
                    pts_mm,
                    dxfattribs={"layer": layer, "closed": True},
                )
                count += 1
        return count

    n_outer = add_contours(outer_list, "OUTER_CUT")
    n_inner = add_contours(inner_list, "INNER_CUT")

    doc.saveas(out_path)
    print(f"[OK] Saved {out_path}  ({n_outer} outer, {n_inner} inner polylines)")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DEBUG PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

def save_debug(
    bgr: np.ndarray,
    outer_list: list[list[np.ndarray]],
    inner_list: list[list[np.ndarray]],
    out_path: str,
) -> None:
    vis = bgr.copy()
    for cnts in outer_list:
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)   # red = outer
    for cnts in inner_list:
        cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)   # blue = inner
    cv2.imwrite(out_path, vis)
    print(f"[DEBUG] Preview saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert image/PDF to DXF cut file")
    ap.add_argument("input",         help="Input file (JPG, PNG, BMP, TIFF, PDF)")
    ap.add_argument("-o", "--output",help="Output DXF path (default: same name + .dxf)")
    ap.add_argument("--dpi",         type=int,   default=300,  help="DPI for PDF / scale reference (default 300)")
    ap.add_argument("--page",        type=int,   default=1,    help="PDF page number 1-based (default 1)")
    ap.add_argument("--width-mm",    type=float, default=None, help="Physical output width in mm")
    ap.add_argument("--epsilon",     type=float, default=1.0,  help="Smoothing factor 0.1–5.0 (default 1.0)")
    ap.add_argument("--min-area",    type=int,   default=50,   help="Min contour area in pixels (default 50)")
    ap.add_argument("--smooth",      type=int,   default=3,    help="Chaikin iterations (default 3, 0=off)")
    ap.add_argument("--threshold",   type=int,   default=None, help="Manual threshold 0–255 (default: auto Otsu)")
    ap.add_argument("--invert",      action="store_true",  default=None, help="Force invert image")
    ap.add_argument("--no-invert",   action="store_true",  help="Disable auto-invert")
    ap.add_argument("--no-upscale",  action="store_true",  help="Disable auto-upscale of small images")
    ap.add_argument("--debug",       action="store_true",  help="Save debug PNG showing detected contours")
    args = ap.parse_args()

    # Resolve force_invert flag
    if args.no_invert:
        force_invert = False
    elif args.invert:
        force_invert = True
    else:
        force_invert = None  # auto-detect

    # Output path
    out_path = args.output or str(Path(args.input).with_suffix(".dxf"))

    # ── Load ──────────────────────────────────────────────────────────────────
    bgr, mm_per_pixel = load_input(args.input, args.dpi, args.page)
    img_h, img_w      = bgr.shape[:2]
    print(f"[INFO] Loaded {img_w}x{img_h}  ({mm_per_pixel*img_w:.0f}x{mm_per_pixel*img_h:.0f} mm at {args.dpi} dpi)")

    # ── Auto-upscale ──────────────────────────────────────────────────────────
    if not args.no_upscale:
        bgr      = _upscale_bgr(bgr)
        img_h, img_w = bgr.shape[:2]

    # ── Override mm_per_pixel if --width-mm given ─────────────────────────────
    if args.width_mm:
        mm_per_pixel = args.width_mm / img_w
        print(f"[INFO] Scale set by --width-mm {args.width_mm}: {mm_per_pixel:.4f} mm/px")

    # ── Colour-aware separation ───────────────────────────────────────────────
    color_masks = _smart_gray(bgr)

    outer_list: list[list[np.ndarray]] = []
    inner_list: list[list[np.ndarray]] = []

    if color_masks:
        # Multi-colour path: trace each colour cluster independently
        for mask in color_masks:
            o, i = find_and_classify(mask, args.min_area)
            if o or i:
                outer_list.append(o)
                inner_list.append(i)
    else:
        # Simple path: standard grayscale
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        binary = preprocess(gray, args.threshold, force_invert)
        o, i   = find_and_classify(binary, args.min_area)
        outer_list.append(o)
        inner_list.append(i)

    total_outer = sum(len(x) for x in outer_list)
    total_inner = sum(len(x) for x in inner_list)
    print(f"[INFO] Contours found: {total_outer} outer, {total_inner} inner")

    if total_outer == 0 and total_inner == 0:
        print("[WARN] No contours detected. Try --debug to inspect, or adjust --threshold / --invert.")

    # ── Debug preview ─────────────────────────────────────────────────────────
    if args.debug:
        debug_path = str(Path(out_path).with_name(Path(args.input).stem + "_debug.png"))
        save_debug(bgr, outer_list, inner_list, debug_path)

    # ── Write DXF ─────────────────────────────────────────────────────────────
    write_dxf(outer_list, inner_list, mm_per_pixel, img_h, out_path, args.epsilon, args.smooth)


if __name__ == "__main__":
    main()
