#!/usr/bin/env python3
"""
img_2_dxf_high_res.py  --  High-resolution image/PDF to DXF for sign manufacturing.

Key improvements over image_to_dxf.py:
  - Upscales to 4000 px long side (captures thin strokes)
  - K-Means used to REMOVE background, then all foreground combined into one mask
    -> eliminates double-lines from anti-aliasing / colour gradients
  - Stroke-fill kernel scales with image resolution
  - CLAHE contrast enhancement for low-contrast images

Output layers:
  OUTER_CUT (red)  -- outer perimeters
  INNER_CUT (blue) -- holes / counters (inside O, D, A, B, etc.)

Usage:
  python img_2_dxf_high_res.py input.jpg --width-mm 400
  python img_2_dxf_high_res.py logo.pdf  --width-mm 600 --debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import ezdxf

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ==============================================================================
# 1.  LOAD
# ==============================================================================

def load_input(path: str, dpi: int, page: int) -> tuple[np.ndarray, float]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    ext = p.suffix.lower()

    if ext == ".pdf":
        if not HAS_FITZ:
            sys.exit("[ERROR] PDF needs pymupdf. Run: pip install pymupdf")
        doc = fitz.open(str(p))
        idx = page - 1
        if idx >= len(doc):
            sys.exit(f"[ERROR] PDF has {len(doc)} page(s); --page {page} out of range.")
        pg  = doc[idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = pg.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
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


# ==============================================================================
# 2.  HIGH-RES UPSCALE  (target 4000 px on long side)
# ==============================================================================

TARGET_LONG_SIDE = 4000

def upscale_high_res(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    long = max(h, w)
    if long >= TARGET_LONG_SIDE:
        print(f"[INFO] Image already {w}x{h}, no upscale needed")
        return bgr

    scale = TARGET_LONG_SIDE / long
    new_w = int(w * scale)
    new_h = int(h * scale)
    up    = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Two-pass unsharp mask for crisp edges
    for sigma, strength in [(1.5, 0.4), (3.0, 0.25)]:
        blur = cv2.GaussianBlur(up, (0, 0), sigma)
        up   = cv2.addWeighted(up, 1.0 + strength, blur, -strength, 0)

    print(f"[INFO] Upscaled {w}x{h} -> {new_w}x{new_h}  (x{scale:.1f})")
    return up


# ==============================================================================
# 3.  BACKGROUND REMOVAL VIA K-MEANS
#
#     Strategy: run K-Means to find colour clusters, label any near-white or
#     near-image-edge cluster as "background", then combine ALL other pixels
#     into a single foreground mask.  This avoids double-lines from gradients
#     and anti-aliasing because every foreground pixel ends up in ONE mask.
# ==============================================================================

def _stroke_kernel(img_h: int, img_w: int, scale: float = 1.0) -> np.ndarray:
    """Morphological kernel sized proportionally to image resolution."""
    long = max(img_h, img_w)
    size = max(3, int(round(long / 350 * scale)))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _has_light_background(bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw   = max(1, int(min(h, w) * 0.08))
    border = np.concatenate([
        gray[:bw, :].flatten(), gray[-bw:, :].flatten(),
        gray[:, :bw].flatten(), gray[:, -bw:].flatten(),
    ])
    return border.mean() > 128


def build_foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Use K-Means in Lab colour space to identify background pixels, then
    return a single binary mask where ALL non-background pixels = 255.

    This is the core fix:
      - Gradient text (e.g. matplotlib blue->dark) -> all shades combined = solid
      - Anti-aliasing halo -> classified as near-background, excluded
      - Multi-colour logos -> all colours combined into one solid silhouette
    """
    img_h, img_w = bgr.shape[:2]

    # If background is dark, invert first so bg detection logic stays the same
    light_bg = _has_light_background(bgr)
    if not light_bg:
        print("[INFO] Dark background detected -- auto-inverting")
        bgr = cv2.bitwise_not(bgr)

    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    # Sample every 3rd pixel to speed up K-Means on large images
    step   = 3
    sample = lab[::step, ::step].reshape(-1, 3).astype(np.float32)

    k        = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, _, centers = cv2.kmeans(sample, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    # Assign every pixel to nearest center
    pixels = lab.reshape(-1, 3).astype(np.float32)
    dists  = np.stack([
        np.sum((pixels - centers[ci]) ** 2, axis=1)
        for ci in range(k)
    ])  # shape (k, N)
    labels = np.argmin(dists, axis=0).reshape(img_h, img_w)

    # Label background clusters:
    #   1. Near-white: L > 180 AND chroma < 30
    #   2. Near-border: cluster whose pixels are concentrated at image edges
    bg_clusters: set[int] = set()
    h_px, w_px = img_h, img_w
    bw = max(1, int(min(h_px, w_px) * 0.08))

    border_mask = np.zeros((img_h, img_w), dtype=bool)
    border_mask[:bw, :]  = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw]  = True
    border_mask[:, -bw:] = True

    for ci in range(k):
        L, a, b_val = centers[ci]
        chroma = float(np.sqrt((float(a) - 128) ** 2 + (float(b_val) - 128) ** 2))

        # Near-white cluster
        if L > 180 and chroma < 30:
            bg_clusters.add(ci)
            continue

        # Cluster strongly concentrated in the border region = background
        cluster_pixels  = int(np.sum(labels == ci))
        border_pixels   = int(np.sum((labels == ci) & border_mask))
        if cluster_pixels > 0:
            border_ratio = border_pixels / cluster_pixels
            total_ratio  = cluster_pixels / (img_h * img_w)
            # If >60% of this cluster's pixels are in the border AND
            # the cluster covers >5% of image area, it is the background
            if border_ratio > 0.60 and total_ratio > 0.05:
                bg_clusters.add(ci)

    print(f"[INFO] K-Means K={k}: {k - len(bg_clusters)} foreground cluster(s), "
          f"{len(bg_clusters)} background cluster(s)")

    # Combine all foreground pixels into one mask
    fg = np.zeros((img_h, img_w), dtype=np.uint8)
    for ci in range(k):
        if ci not in bg_clusters:
            fg[labels == ci] = 255

    # Morphological closing: fills thin stroke gaps & merges anti-aliasing halos
    sk = _stroke_kernel(img_h, img_w)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, sk)

    # Remove specks — minimum area scales with resolution
    min_cc = max(100, int((img_h * img_w) * 0.000035))
    n_lbl, cc_lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    for lbl in range(1, n_lbl):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cc:
            clean[cc_lbl == lbl] = 255

    return clean


# ==============================================================================
# 4.  PLAIN GRAYSCALE PATH  (dark-on-white or manual threshold)
# ==============================================================================

def _fix_orientation(binary: np.ndarray) -> np.ndarray:
    h, w    = binary.shape
    px      = int(min(h, w) * 0.05) + 1
    corners = [binary[:px, :px], binary[:px, -px:], binary[-px:, :px], binary[-px:, -px:]]
    if float(np.concatenate([c.flatten() for c in corners]).mean()) > 128:
        return cv2.bitwise_not(binary)
    return binary


def preprocess_gray(gray: np.ndarray, threshold: int | None, force_invert) -> np.ndarray:
    h, w = gray.shape
    bw   = max(1, int(min(h, w) * 0.08))
    border = np.concatenate([
        gray[:bw, :].flatten(), gray[-bw:, :].flatten(),
        gray[:, :bw].flatten(), gray[:, -bw:].flatten(),
    ])
    dark_bg = border.mean() < 128

    if force_invert is True:
        gray = cv2.bitwise_not(gray)
    elif force_invert is None and dark_bg:
        print("[INFO] Dark background -- auto-inverting")
        gray = cv2.bitwise_not(gray)

    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    if threshold is not None:
        _, binary = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio  = float(binary.mean()) / 255.0
        if fg_ratio > 0.85 or fg_ratio < 0.02:
            print("[INFO] Low contrast -- using adaptive threshold")
            binary = cv2.adaptiveThreshold(
                filtered, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                blockSize=31, C=10
            )

    sk     = _stroke_kernel(h, w)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, sk)
    binary = _fix_orientation(binary)
    return binary


# ==============================================================================
# 5.  FIND & CLASSIFY CONTOURS
# ==============================================================================

def find_and_classify(
    binary: np.ndarray,
    min_area: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    if hier is None:
        return [], []
    hier = hier[0]

    def depth(idx: int) -> int:
        d = 0
        while hier[idx][3] != -1:
            idx = hier[idx][3]
            d  += 1
        return d

    outer, inner = [], []
    for i, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) < min_area:
            continue
        (outer if depth(i) % 2 == 0 else inner).append(cnt)
    return outer, inner


# ==============================================================================
# 6.  SMOOTH CONTOUR  (Douglas-Peucker + Chaikin)
# ==============================================================================

def smooth_contour(cnt: np.ndarray, epsilon: float, smooth: int) -> np.ndarray:
    arc        = cv2.arcLength(cnt, closed=True)
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


# ==============================================================================
# 7.  WRITE DXF
# ==============================================================================

def write_dxf(
    outer: list[np.ndarray],
    inner: list[np.ndarray],
    mm_per_pixel: float,
    img_height: int,
    out_path: str,
    epsilon: float,
    smooth: int,
) -> None:
    doc       = ezdxf.new("R2010")
    doc.units = 4  # mm
    msp       = doc.modelspace()
    doc.layers.add("OUTER_CUT", color=1)  # red
    doc.layers.add("INNER_CUT", color=5)  # blue

    def add(cnts: list[np.ndarray], layer: str) -> int:
        n = 0
        for cnt in cnts:
            pts_px = smooth_contour(cnt, epsilon, smooth)
            if len(pts_px) < 3:
                continue
            pts_mm = [
                (float(x) * mm_per_pixel, float(img_height - y) * mm_per_pixel)
                for x, y in pts_px
            ]
            msp.add_lwpolyline(pts_mm, dxfattribs={"layer": layer, "closed": True})
            n += 1
        return n

    n_o = add(outer, "OUTER_CUT")
    n_i = add(inner, "INNER_CUT")
    doc.saveas(out_path)
    print(f"[OK] Saved {out_path}  ({n_o} outer, {n_i} inner polylines)")


# ==============================================================================
# 8.  DEBUG PREVIEW
# ==============================================================================

def save_debug(
    bgr: np.ndarray,
    fg_mask: np.ndarray,
    outer: list[np.ndarray],
    inner: list[np.ndarray],
    out_path: str,
) -> None:
    # Save the combined foreground mask (what gets traced)
    mask_path = str(Path(out_path).with_name(Path(out_path).stem + "_mask.png"))
    cv2.imwrite(mask_path, fg_mask)
    print(f"[DEBUG] Foreground mask: {mask_path}")

    vis       = bgr.copy()
    thickness = max(2, int(max(bgr.shape[:2]) / 1000))

    for cnt in outer:
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.drawContours(vis, [cnt], -1, color, thickness)

    for cnt in inner:
        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
        cv2.drawContours(vis, [cnt], -1, color, thickness)

    # Legend
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = max(0.5, vis.shape[1] / 2500)
    h_vis   = vis.shape[0]
    cv2.putText(vis, f"OUTER: {len(outer)} shapes (each = different colour)",
                (10, h_vis - 40), font, scale, (255, 255, 255), 2)
    cv2.putText(vis, f"INNER holes: {len(inner)}",
                (10, h_vis - 14), font, scale, (200, 200, 255), 2)

    max_side = 2000
    h, w     = vis.shape[:2]
    if max(h, w) > max_side:
        s   = max_side / max(h, w)
        vis = cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    cv2.imwrite(out_path, vis)
    print(f"[DEBUG] Contour preview: {out_path}")


# ==============================================================================
# 9.  MAIN
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="High-res image/PDF to DXF -- clean single-line traces for sign cutting"
    )
    ap.add_argument("input",          help="Input file (JPG, PNG, BMP, TIFF, PDF)")
    ap.add_argument("-o", "--output", help="Output DXF path (default: same name + .dxf)")
    ap.add_argument("--dpi",          type=int,   default=300,  help="DPI for PDF input (default 300)")
    ap.add_argument("--page",         type=int,   default=1,    help="PDF page 1-based (default 1)")
    ap.add_argument("--width-mm",     type=float, default=None, help="Physical output width in mm")
    ap.add_argument("--epsilon",      type=float, default=0.8,  help="Simplify factor 0.1-5.0 (default 0.8)")
    ap.add_argument("--min-area",     type=int,   default=80,   help="Min contour area px (default 80)")
    ap.add_argument("--smooth",       type=int,   default=3,    help="Chaikin iterations (default 3)")
    ap.add_argument("--threshold",    type=int,   default=None, help="Manual threshold 0-255 (default: auto)")
    ap.add_argument("--invert",       action="store_true", help="Force invert image")
    ap.add_argument("--no-invert",    action="store_true", help="Disable auto-invert")
    ap.add_argument("--no-upscale",   action="store_true", help="Skip high-res upscale")
    ap.add_argument("--gray",         action="store_true", help="Force plain grayscale (skip K-Means)")
    ap.add_argument("--debug",        action="store_true", help="Save debug PNG + foreground mask")
    args = ap.parse_args()

    force_invert = False if args.no_invert else (True if args.invert else None)
    out_path     = args.output or str(Path(args.input).with_suffix(".dxf"))

    # Load
    bgr, mm_per_pixel = load_input(args.input, args.dpi, args.page)
    img_h, img_w      = bgr.shape[:2]
    print(f"[INFO] Loaded {img_w}x{img_h}")

    # High-res upscale
    if not args.no_upscale:
        bgr      = upscale_high_res(bgr)
        img_h, img_w = bgr.shape[:2]

    # Scale
    if args.width_mm:
        mm_per_pixel = args.width_mm / img_w
        print(f"[INFO] Scale: {mm_per_pixel:.5f} mm/px  ({args.width_mm} mm wide)")
    else:
        print(f"[INFO] Scale: {mm_per_pixel:.5f} mm/px  ({mm_per_pixel*img_w:.0f} x {mm_per_pixel*img_h:.0f} mm)")

    # Min area scales with resolution
    effective_min_area = max(80, int(args.min_area * (img_w / 1300) ** 2))
    print(f"[INFO] Effective min-area: {effective_min_area} px^2")

    # Build foreground mask
    if args.gray:
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fg     = preprocess_gray(gray, args.threshold, force_invert)
    else:
        fg = build_foreground_mask(bgr)

    # Find contours
    outer, inner = find_and_classify(fg, effective_min_area)
    print(f"[INFO] Contours: {len(outer)} outer, {len(inner)} inner")

    if len(outer) == 0 and len(inner) == 0:
        print("[WARN] No contours found. Try --debug to inspect the mask, or add --gray.")

    # Debug
    if args.debug:
        debug_path = str(Path(out_path).with_name(Path(args.input).stem + "_hires_debug.png"))
        save_debug(bgr, fg, outer, inner, debug_path)

    # Write DXF
    write_dxf(outer, inner, mm_per_pixel, img_h, out_path, args.epsilon, args.smooth)


if __name__ == "__main__":
    main()
