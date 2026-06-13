"""
Exam4 — SIFT Feature Matching & Panorama Stitching
==================================================
  Họ và tên   : Huỳnh Phát Lợi
  MSHV        : KHMT836016
  Môn         : Computer Vision
  Bài tập ngày: 13/06/2026

Task (exercise.png):
  1. Capture a sequence of overlapping photos (>= 1/3 overlap) around a place.
  2. Use SIFT to extract keypoints and match corresponding keypoints between
     each pair of images, visualizing the matches.
  3. Stitch the images into a panorama of the whole scene.

Input  : exam4/img/IMG_2524.JPG, IMG_2525.JPG, IMG_2526.JPG  (left -> right)
Output : exam4/outputs/*.png

Run:
    python3 exam4/image_stitching_sift.py
"""

import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

# ── Configuration ──────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

# Input/output folders are configurable so the same pipeline can run on several
# photo sets, e.g.  `python3 image_stitching_sift.py img2 outputs2`.
# Defaults reproduce the first set (img/ -> outputs/).
DEFAULT_IMG_DIR = "img"
DEFAULT_OUT_DIR = "outputs"

MAX_DIM = 1500          # resize so the long side is at most this many pixels
RATIO = 0.75            # Lowe's ratio-test threshold
RANSAC_THRESH = 5.0     # reprojection error (px) for findHomography RANSAC


# ── 2. Load + orient + resize ──────────────────────────────────────────────
def load_image(path):
    """Open an image, apply EXIF orientation, resize, return (color_bgr, gray)."""
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = pil.size
    scale = MAX_DIM / float(max(w, h))
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return bgr, gray


# ── 3. SIFT keypoints ──────────────────────────────────────────────────────
def detect(sift, gray):
    """Detect SIFT keypoints and compute descriptors."""
    return sift.detectAndCompute(gray, None)


# ── 4. Matching (Lowe ratio test) ──────────────────────────────────────────
def match(desc_a, desc_b, ratio=RATIO):
    """knn match A->B and keep good matches via Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


def draw_and_save_matches(img_a, kp_a, img_b, kp_b, matches, mask, out_path):
    """Visualize correspondences; if a mask is given, draw RANSAC inliers only."""
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if mask is not None:
        draw_params["matchesMask"] = mask.ravel().tolist()
    vis = cv2.drawMatches(img_a, kp_a, img_b, kp_b, matches, None, **draw_params)
    cv2.imwrite(out_path, vis)
    return vis


# ── 5. Homography ──────────────────────────────────────────────────────────
def homography(kp_a, kp_b, matches):
    """Estimate homography H mapping image A points onto image B (RANSAC)."""
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_THRESH)
    return H, mask


# ── 6. Manual stitching ────────────────────────────────────────────────────
def _warped_corners(shape, H):
    h, w = shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H)


def stitch_three(left, center, right, H_left, H_right):
    """
    Warp `left` and `right` into the plane of `center` and composite them on a
    canvas large enough to hold all three. H_left maps left->center,
    H_right maps right->center.
    """
    # Collect every corner in the center's coordinate frame to size the canvas.
    all_corners = np.concatenate([
        _warped_corners(left.shape, H_left),
        _warped_corners(center.shape, np.eye(3)),
        _warped_corners(right.shape, H_right),
    ])
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation so the whole panorama has positive coordinates.
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    size = (x_max - x_min, y_max - y_min)

    pano = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Paint order: outer images first, then center on top (sharpest anchor).
    for img, H in [(left, H_left), (right, H_right), (center, np.eye(3))]:
        warped = cv2.warpPerspective(img, T @ H, size)
        mask = warped.sum(axis=2) > 0
        pano[mask] = warped[mask]
    return pano


def crop_black(img):
    """Crop the largest axis-aligned box containing non-black pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ── 7. Bonus: OpenCV built-in Stitcher ─────────────────────────────────────
def opencv_stitch(images_bgr):
    # Try PANORAMA mode first; fall back to SCANS (affine) mode, which is more
    # robust for a small set of images with strong perspective change where
    # PANORAMA's bundle adjustment can fail (ERR_CAMERA_PARAMS_ADJUST_FAIL).
    for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
        stitcher = cv2.Stitcher_create(mode)
        stitcher.setPanoConfidenceThresh(0.3)
        status, pano = stitcher.stitch(images_bgr)
        if status == cv2.Stitcher_OK:
            return status, pano
    return status, None


# ── Ordering ───────────────────────────────────────────────────────────────
def order_left_to_right(descs):
    """
    For 3 images, pick the center as the one with the most total matches to the
    others, then return indices [left, center, right]. This avoids hard-coding
    a left->right order per photo set.
    """
    n = len(descs)
    counts = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = len(match(descs[i], descs[j]))
            counts[i][j] = counts[j][i] = c
    center = max(range(n), key=lambda i: sum(counts[i]))
    others = [i for i in range(n) if i != center]
    return [others[0], center, others[1]]


# ── 8. Main pipeline ───────────────────────────────────────────────────────
def main(img_dir=DEFAULT_IMG_DIR, out_dir=DEFAULT_OUT_DIR):
    IMG_DIR = img_dir if os.path.isabs(img_dir) else os.path.join(HERE, img_dir)
    OUT_DIR = out_dir if os.path.isabs(out_dir) else os.path.join(HERE, out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    sift = cv2.SIFT_create()
    saved = []

    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.JPG")) +
                   glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if len(files) != 3:
        raise SystemExit("Expected 3 images in %s, found %d" % (IMG_DIR, len(files)))

    print("Loading images from %s (EXIF-corrected, resized to <= %dpx)..."
          % (os.path.relpath(IMG_DIR, HERE), MAX_DIM))
    colors, grays, names = [], [], []
    for f in files:
        bgr, gray = load_image(f)
        colors.append(bgr)
        grays.append(gray)
        names.append(os.path.splitext(os.path.basename(f))[0])
        print("  %-14s -> %dx%d" % (os.path.basename(f), bgr.shape[1], bgr.shape[0]))

    # Step 3 — keypoints
    print("\nDetecting SIFT keypoints...")
    kps, descs = [], []
    for bgr, gray, name in zip(colors, grays, names):
        kp, desc = detect(sift, gray)
        kps.append(kp)
        descs.append(desc)
        vis = cv2.drawKeypoints(
            bgr, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        p = os.path.join(OUT_DIR, "keypoints_%s.png" % name)
        cv2.imwrite(p, vis)
        saved.append(p)
        print("  %-14s -> %d keypoints" % (name, len(kp)))

    # Determine left -> center -> right order from pairwise match counts.
    order = order_left_to_right(descs)
    colors = [colors[i] for i in order]
    grays = [grays[i] for i in order]
    names = [names[i] for i in order]
    kps = [kps[i] for i in order]
    descs = [descs[i] for i in order]
    print("\nStitching order (left -> right): %s" % " -> ".join(names))

    # Steps 4 & 5 — matching + homography for adjacent pairs (left|center, center|right)
    pairs = [(0, 1), (1, 2)]
    homographies = {}  # (a,b) -> H mapping a -> b
    print("\nMatching adjacent pairs and estimating homographies...")
    for a, b in pairs:
        good = match(descs[a], descs[b])
        H, mask = homography(kps[a], kps[b], good)
        inliers = int(mask.sum()) if mask is not None else 0
        homographies[(a, b)] = H
        print("  %s <-> %s : %d good matches, %d RANSAC inliers"
              % (names[a], names[b], len(good), inliers))

        # All good matches.
        p = os.path.join(OUT_DIR, "matches_%s_%s.png" % (names[a], names[b]))
        draw_and_save_matches(colors[a], kps[a], colors[b], kps[b], good, None, p)
        saved.append(p)
        # Inliers only.
        p_in = os.path.join(OUT_DIR, "matches_inliers_%s_%s.png" % (names[a], names[b]))
        draw_and_save_matches(colors[a], kps[a], colors[b], kps[b], good, mask, p_in)
        saved.append(p_in)

    # Step 6 — manual stitching, anchored on the center image (index 1).
    print("\nStitching (manual, anchored on center image)...")
    H_left = homographies[(0, 1)]   # left  -> center
    H_right = np.linalg.inv(homographies[(1, 2)])  # right -> center
    pano = stitch_three(colors[0], colors[1], colors[2], H_left, H_right)
    pano = crop_black(pano)
    p = os.path.join(OUT_DIR, "panorama_manual.png")
    cv2.imwrite(p, pano)
    saved.append(p)
    print("  panorama_manual.png -> %dx%d" % (pano.shape[1], pano.shape[0]))

    # Step 7 — OpenCV Stitcher comparison.
    print("\nStitching (OpenCV cv2.Stitcher, bonus comparison)...")
    status, pano_cv = opencv_stitch(colors)
    if status == cv2.Stitcher_OK:
        p = os.path.join(OUT_DIR, "panorama_opencv_stitcher.png")
        cv2.imwrite(p, pano_cv)
        saved.append(p)
        print("  OK -> panorama_opencv_stitcher.png (%dx%d)"
              % (pano_cv.shape[1], pano_cv.shape[0]))
    else:
        print("  cv2.Stitcher returned non-OK status: %d (skipped)" % status)

    print("\nDone. Saved %d files to %s:" % (len(saved), OUT_DIR))
    for p in saved:
        print("  -", os.path.relpath(p, HERE))


if __name__ == "__main__":
    # Usage: python3 image_stitching_sift.py [IMG_DIR] [OUT_DIR]
    args = sys.argv[1:]
    img_dir = args[0] if len(args) > 0 else DEFAULT_IMG_DIR
    out_dir = args[1] if len(args) > 1 else DEFAULT_OUT_DIR
    main(img_dir, out_dir)
