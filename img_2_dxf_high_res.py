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
# ------------------------------------------------------------------------------
# Turn whatever the user gave us (an image file OR a PDF) into a pixel picture we
# can process, and work out how many millimetres each pixel represents.
#   - PDF  -> rendered to pixels at the chosen DPI using PyMuPDF (fitz).
#   - Image-> read with OpenCV; if OpenCV can't (e.g. CMYK/RGBA), fall back to Pillow.
# Returns: (bgr image as a NumPy array, millimetres-per-pixel).
# ==============================================================================

def load_input(path: str, dpi: int, page: int) -> tuple[np.ndarray, float]:
    """Load an image or one PDF page into a BGR array + its mm-per-pixel scale."""
    p = Path(path)
    # Fail early with a clear message if the file isn't there.
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    ext = p.suffix.lower()

    if ext == ".pdf":
        # --- PDF branch: render the chosen page to a raster image at `dpi` ---
        if not HAS_FITZ:
            sys.exit("[ERROR] PDF needs pymupdf. Run: pip install pymupdf")
        doc = fitz.open(str(p))
        idx = page - 1                                   # CLI page is 1-based; fitz is 0-based
        if idx >= len(doc):
            sys.exit(f"[ERROR] PDF has {len(doc)} page(s); --page {page} out of range.")
        pg  = doc[idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)            # 72 = PDF's native points-per-inch
        pix = pg.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        # raw bytes -> NumPy (H, W, 3); fitz gives RGB, OpenCV wants BGR
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        # --- Image branch: let OpenCV read it; if it returns None, use Pillow ---
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            # Pillow handles formats/colour-modes OpenCV chokes on (CMYK, RGBA, ...)
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

    # mm-per-pixel from DPI (25.4 mm per inch). NOTE: this is a default — if the user
    # passes --width-mm, main() overrides it with the exact requested width.
    mm_per_pixel = 25.4 / dpi
    return bgr, mm_per_pixel


# ==============================================================================
# 2.  HIGH-RES UPSCALE  (target 4000 px on long side)
# ------------------------------------------------------------------------------
# Small/low-res art loses thin strokes when traced. We enlarge the picture to
# ~4000 px on its long edge (smooth Lanczos resize) and then sharpen it, so the
# letter edges stay crisp and don't disappear during the black/white step.
# ==============================================================================

TARGET_LONG_SIDE = 4000

def upscale_high_res(bgr: np.ndarray) -> np.ndarray:
    """Enlarge to ~4000 px long side + sharpen; returns the image unchanged if already big."""
    h, w = bgr.shape[:2]
    long = max(h, w)
    if long >= TARGET_LONG_SIDE:
        print(f"[INFO] Image already {w}x{h}, no upscale needed")
        return bgr

    scale = TARGET_LONG_SIDE / long                      # how much we need to enlarge
    new_w = int(w * scale)
    new_h = int(h * scale)
    up    = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)  # high-quality resize

    # Two-pass unsharp mask: blur, then subtract the blur to exaggerate edges.
    # Two passes (fine + coarse) sharpen both small details and broad edges.
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
    """True if the image's BORDER is bright -> it's art on a light background.
    We look only at an 8%-wide frame around the edges (the subject is usually in
    the middle, so the border is a good sample of the background)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw   = max(1, int(min(h, w) * 0.08))                 # border band width = 8% of the short side
    border = np.concatenate([
        gray[:bw, :].flatten(), gray[-bw:, :].flatten(),  # top + bottom bands
        gray[:, :bw].flatten(), gray[:, -bw:].flatten(),  # left + right bands
    ])
    return border.mean() > 128                            # >128 (mid-grey) = light


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

    # Work in Lab colour space (L=lightness, a/b=colour) — distances there match how
    # the eye groups colours, so clustering is more reliable than in RGB.
    lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    # Sample every 3rd pixel to speed up K-Means on large images (we only need the
    # cluster CENTRES; we'll classify all pixels against them afterwards).
    step   = 3
    sample = lab[::step, ::step].reshape(-1, 3).astype(np.float32)

    # Group the colours into k=8 clusters (centres = the 8 dominant colours).
    k        = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, _, centers = cv2.kmeans(sample, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    # Label EVERY pixel with its nearest cluster centre (full-resolution this time).
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

    # Every pixel NOT in a background cluster becomes white (255) in one solid mask.
    fg = np.zeros((img_h, img_w), dtype=np.uint8)
    for ci in range(k):
        if ci not in bg_clusters:
            fg[labels == ci] = 255

    # Closing = dilate then erode: fills tiny gaps in strokes and welds anti-aliasing
    # halos onto the letter, so each shape traces as one clean outline.
    sk = _stroke_kernel(img_h, img_w)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, sk)

    # Drop tiny blobs (dust/noise). The size threshold scales with the image area so
    # it behaves the same on small and huge images.
    min_cc = max(100, int((img_h * img_w) * 0.000035))
    n_lbl, cc_lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    for lbl in range(1, n_lbl):                          # label 0 is the background
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cc:
            clean[cc_lbl == lbl] = 255

    return clean


# ==============================================================================
# 4.  PLAIN GRAYSCALE PATH  (dark-on-white or manual threshold)
# ------------------------------------------------------------------------------
# Simpler alternative to K-Means, used when you pass --gray. Good for plain black
# artwork on white: enhance contrast (CLAHE), denoise, then threshold to black/white.
# Produces the same kind of foreground mask that section 5 will trace.
# ==============================================================================

def _fix_orientation(binary: np.ndarray) -> np.ndarray:
    """Make sure the SHAPE is white on black: if the corners are mostly white, the
    image is inverted, so flip it."""
    h, w    = binary.shape
    px      = int(min(h, w) * 0.05) + 1
    corners = [binary[:px, :px], binary[:px, -px:], binary[-px:, :px], binary[-px:, -px:]]
    if float(np.concatenate([c.flatten() for c in corners]).mean()) > 128:
        return cv2.bitwise_not(binary)
    return binary


def preprocess_gray(gray: np.ndarray, threshold: int | None, force_invert) -> np.ndarray:
    """Grayscale -> clean black/white mask: auto-invert dark backgrounds, boost
    contrast, denoise, threshold (manual / Otsu / adaptive), close gaps."""
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

    # CLAHE = local contrast boost (helps faint/low-contrast scans).
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Bilateral filter = smooth noise but keep edges sharp.
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    if threshold is not None:
        # User gave an exact black/white cutoff.
        _, binary = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    else:
        # Otsu picks the cutoff automatically from the histogram.
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio  = float(binary.mean()) / 255.0
        # If Otsu produced almost-all or almost-nothing, the image is low-contrast/uneven
        # -> switch to ADAPTIVE threshold (decides per local region).
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
# ------------------------------------------------------------------------------
# Trace the outlines of the white mask and decide which are OUTER edges and which
# are HOLES. RETR_TREE gives the nesting (who is inside whom); a contour's "depth"
# = how many outlines wrap around it. Even depth (0,2,...) = an outer edge; odd
# depth (1,3,...) = a hole/counter. Tiny contours (< min_area) are ignored.
# ==============================================================================

def find_and_classify(
    binary: np.ndarray,
    min_area: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (outer_contours, inner_hole_contours) from the binary mask."""
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    if hier is None:
        return [], []
    hier = hier[0]                                       # hierarchy: [next, prev, child, parent]

    def depth(idx: int) -> int:
        """How many contours enclose contour `idx` (walk up the parent links)."""
        d = 0
        while hier[idx][3] != -1:                        # [3] = parent index, -1 = top level
            idx = hier[idx][3]
            d  += 1
        return d

    outer, inner = [], []
    for i, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) < min_area:              # skip specks
            continue
        # even depth -> outer perimeter; odd depth -> hole inside a letter
        (outer if depth(i) % 2 == 0 else inner).append(cnt)
    return outer, inner


# ==============================================================================
# 6.  SMOOTH CONTOUR  (Douglas-Peucker + Chaikin)
# ------------------------------------------------------------------------------
# Raw pixel outlines are jagged and have thousands of points. Two steps clean them:
#   1. Douglas-Peucker (cv2.approxPolyDP) -> drop redundant points (`--epsilon`
#      controls how aggressively; bigger = fewer points / more simplification).
#   2. Chaikin -> round the corners by repeatedly cutting each corner (`--smooth`
#      = how many rounding passes). Result: smooth, light curves for cutting.
# ==============================================================================

def smooth_contour(cnt: np.ndarray, epsilon: float, smooth: int) -> np.ndarray:
    """Simplify (Douglas-Peucker) then round (Chaikin) one contour -> Nx2 points."""
    arc        = cv2.arcLength(cnt, closed=True)         # perimeter length
    approx_eps = (epsilon / 1000.0) * arc                # tolerance scales with size
    simplified = cv2.approxPolyDP(cnt, approx_eps, closed=True)
    pts        = simplified.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return pts
    # Chaikin: replace each point with two points 1/4 and 3/4 along to its neighbour,
    # which shaves the corners off; repeating it makes the outline progressively smoother.
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
# ------------------------------------------------------------------------------
# Build the CAD file: a millimetre-unit DXF with two layers — OUTER_CUT (red) for
# perimeters and INNER_CUT (blue) for holes. Each smoothed contour becomes one
# CLOSED polyline. Pixel coords -> mm via mm_per_pixel, and Y is FLIPPED
# (image Y grows downward; CAD Y grows upward) so the result isn't upside-down.
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
    """Write outer + inner contours to a mm-unit DXF on two coloured layers."""
    doc       = ezdxf.new("R2010")
    doc.units = 4  # 4 = millimetres (DXF INSUNITS code)
    msp       = doc.modelspace()
    doc.layers.add("OUTER_CUT", color=1)  # red  = cut the outline
    doc.layers.add("INNER_CUT", color=5)  # blue = cut the holes/counters

    def add(cnts: list[np.ndarray], layer: str) -> int:
        """Smooth each contour, convert px->mm (flip Y), add as a closed polyline."""
        n = 0
        for cnt in cnts:
            pts_px = smooth_contour(cnt, epsilon, smooth)
            if len(pts_px) < 3:
                continue
            pts_mm = [
                # x*scale ; (img_height - y)*scale  <-- the (img_height - y) is the Y flip
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
# 8.  DEBUG PREVIEW   (only when --debug is passed)
# ------------------------------------------------------------------------------
# Saves two pictures to check the result by eye:
#   *_mask.png  -> the exact black/white foreground that got traced
#   the preview -> the original image with every detected contour drawn on top
#                  (each shape a random colour) + a small legend.
# Nothing here affects the DXF; it's purely for inspection.
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
# 9.  MAIN  —  command-line entry point; runs the whole pipeline in order
# ------------------------------------------------------------------------------
# Flow: parse options -> LOAD (1) -> UPSCALE (2) -> set mm scale -> build the
# foreground mask via K-Means (3) or grayscale (4) -> FIND & CLASSIFY (5) ->
# optional DEBUG (8) -> WRITE DXF (7, which calls SMOOTH 6). Each --option below
# tweaks one of those steps.
# ==============================================================================

def main() -> None:
    # ---- Command-line options (run `python img_2_dxf_high_res.py -h` to see them) ----
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

    # Decide inversion: --no-invert wins, else --invert, else None = auto-detect later.
    force_invert = False if args.no_invert else (True if args.invert else None)
    # Default output = input name with .dxf, unless -o given.
    out_path     = args.output or str(Path(args.input).with_suffix(".dxf"))

    # --- Step 1: LOAD the image / render the PDF page ---
    bgr, mm_per_pixel = load_input(args.input, args.dpi, args.page)
    img_h, img_w      = bgr.shape[:2]
    print(f"[INFO] Loaded {img_w}x{img_h}")

    # --- Step 2: UPSCALE for crisp thin strokes (unless --no-upscale) ---
    if not args.no_upscale:
        bgr      = upscale_high_res(bgr)
        img_h, img_w = bgr.shape[:2]

    # --- Set the real-world scale (mm per pixel) ---
    if args.width_mm:
        # User gave the physical width -> derive mm/px so the DXF is exactly that wide.
        mm_per_pixel = args.width_mm / img_w
        print(f"[INFO] Scale: {mm_per_pixel:.5f} mm/px  ({args.width_mm} mm wide)")
    else:
        # Otherwise keep the DPI-based scale from load_input.
        print(f"[INFO] Scale: {mm_per_pixel:.5f} mm/px  ({mm_per_pixel*img_w:.0f} x {mm_per_pixel*img_h:.0f} mm)")

    # Speck filter grows with resolution so behaviour is consistent across image sizes.
    effective_min_area = max(80, int(args.min_area * (img_w / 1300) ** 2))
    print(f"[INFO] Effective min-area: {effective_min_area} px^2")

    # --- Steps 3/4: build the foreground mask (white shape on black) ---
    if args.gray:
        gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fg     = preprocess_gray(gray, args.threshold, force_invert)   # simple grayscale path
    else:
        fg = build_foreground_mask(bgr)                                # K-Means path (default)

    # --- Step 5: trace + classify into outer perimeters and inner holes ---
    outer, inner = find_and_classify(fg, effective_min_area)
    print(f"[INFO] Contours: {len(outer)} outer, {len(inner)} inner")

    if len(outer) == 0 and len(inner) == 0:
        print("[WARN] No contours found. Try --debug to inspect the mask, or add --gray.")

    # --- Step 8: optional debug images ---
    if args.debug:
        debug_path = str(Path(out_path).with_name(Path(args.input).stem + "_hires_debug.png"))
        save_debug(bgr, fg, outer, inner, debug_path)

    # --- Step 7 (+6): smooth and write the final DXF ---
    write_dxf(outer, inner, mm_per_pixel, img_h, out_path, args.epsilon, args.smooth)


if __name__ == "__main__":
    main()
