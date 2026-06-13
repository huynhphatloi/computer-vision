"""
Exam4 — SIFT Feature Matching & Panorama Stitching  (Google Colab version)
==========================================================================
  Họ và tên   : Huỳnh Phát Lợi
  MSHV        : KHMT836016
  Môn         : Computer Vision
  Bài tập ngày: 13/06/2026

Task (exercise.png):
  1. Capture a sequence of overlapping photos (>= 1/3 overlap) around a place.
  2. Use SIFT to extract keypoints and match corresponding keypoints between
     each pair of images, visualizing the matches.
  3. Stitch the images into a panorama of the whole scene.

Everything (cv2, numpy, matplotlib, PIL) is pre-installed in Colab.
"""

# %% ─────────────────────────────────────────────────────────────────────────
# CELL 1 — Imports & configuration
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

MAX_DIM = 1500           # resize so the long side is at most this many pixels
RATIO = 0.75             # Lowe's ratio-test threshold
RANSAC_THRESH = 5.0      # reprojection error (px) for findHomography RANSAC
MAX_CANVAS_PIXELS = 60_000_000   # safety cap for the manual panorama canvas

print("Libraries ready. cv2 =", cv2.__version__)


# %% ─────────────────────────────────────────────────────────────────────────
# CELL 2 — Upload your images (run, then pick 3 overlapping photos)
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import files  # noqa: E402  (Colab-only import)

uploaded = files.upload()
filenames = sorted(uploaded.keys())
print("\nUploaded %d files:" % len(filenames))
for f in filenames:
    print("  -", f)
assert len(filenames) >= 2, "Please upload at least 2 overlapping images."


# %% ─────────────────────────────────────────────────────────────────────────
# CELL 3 — Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def show(img_bgr, title="", figsize=(12, 7)):
    """Display a BGR image inline with matplotlib (converted to RGB)."""
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_image(path):
    """Open an image, apply EXIF orientation, resize; return (color_bgr, gray)."""
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = pil.size
    scale = MAX_DIM / float(max(w, h))
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return bgr, gray


def match(desc_a, desc_b, ratio=RATIO):
    """knn match A->B and keep good matches via Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    return [m for m, n in knn if m.distance < ratio * n.distance]


def homography(kp_a, kp_b, matches):
    """Estimate homography H mapping image A points onto image B (RANSAC)."""
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_THRESH)


def order_left_to_right(descs):
    """Pick the center image (most total matches) and order [left, center, right]."""
    n = len(descs)
    counts = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = len(match(descs[i], descs[j]))
            counts[i][j] = counts[j][i] = c
    center = max(range(n), key=lambda i: sum(counts[i]))
    others = [i for i in range(n) if i != center]
    return [others[0], center, others[1]] if n == 3 else list(range(n))


def _warped_corners(shape, H):
    h, w = shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H)


def stitch_three(left, center, right, H_left, H_right):
    """Warp left & right into center's plane and composite on a single canvas."""
    all_corners = np.concatenate([
        _warped_corners(left.shape, H_left),
        _warped_corners(center.shape, np.eye(3)),
        _warped_corners(right.shape, H_right),
    ])
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    size = (x_max - x_min, y_max - y_min)

    # Safety guard: a near-degenerate homography can blow the canvas up to
    # billions of pixels and crash the runtime. Bail out gracefully instead.
    if size[0] <= 0 or size[1] <= 0 or size[0] * size[1] > MAX_CANVAS_PIXELS:
        print("  [skip] manual canvas too large (%dx%d) — perspective too "
              "extreme for a single-plane warp; use the OpenCV panorama below."
              % (size[0], size[1]))
        return None

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    pano = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for img, H in [(left, H_left), (right, H_right), (center, np.eye(3))]:
        warped = cv2.warpPerspective(img, T @ H, size)
        mask = warped.sum(axis=2) > 0
        pano[mask] = warped[mask]
    return pano


def crop_black(img):
    """Crop the bounding box of non-black pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def opencv_stitch(images_bgr):
    """OpenCV's built-in stitcher: PANORAMA mode, falling back to SCANS."""
    status = None
    for mode in (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS):
        stitcher = cv2.Stitcher_create(mode)
        stitcher.setPanoConfidenceThresh(0.3)
        status, pano = stitcher.stitch(images_bgr)
        if status == cv2.Stitcher_OK:
            return status, pano
    return status, None


print("Helpers defined.")


# %% ─────────────────────────────────────────────────────────────────────────
# CELL 4 — Load images + detect SIFT keypoints
# ─────────────────────────────────────────────────────────────────────────────
sift = cv2.SIFT_create()
colors, grays, names, kps, descs = [], [], [], [], []

for f in filenames:
    bgr, gray = load_image(f)
    kp, desc = sift.detectAndCompute(gray, None)
    colors.append(bgr); grays.append(gray); names.append(f)
    kps.append(kp); descs.append(desc)
    vis = cv2.drawKeypoints(bgr, kp, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("%-20s -> %d keypoints" % (f, len(kp)))
    show(vis, "SIFT keypoints — %s (%d)" % (f, len(kp)))


# %% ─────────────────────────────────────────────────────────────────────────
# CELL 5 — Order images, then match + visualize each adjacent pair
# ─────────────────────────────────────────────────────────────────────────────
order = order_left_to_right(descs)
colors = [colors[i] for i in order]; grays = [grays[i] for i in order]
names = [names[i] for i in order]; kps = [kps[i] for i in order]
descs = [descs[i] for i in order]
print("Stitching order (left -> right):", " -> ".join(names))

homographies = {}   # (a, b) -> H mapping image a -> image b
for a, b in zip(range(len(names) - 1), range(1, len(names))):
    good = match(descs[a], descs[b])
    H, mask = homography(kps[a], kps[b], good)
    homographies[(a, b)] = H
    inliers = int(mask.sum()) if mask is not None else 0
    print("%s <-> %s : %d good matches, %d RANSAC inliers"
          % (names[a], names[b], len(good), inliers))

    # All good matches (Lowe ratio test).
    vis_all = cv2.drawMatches(
        colors[a], kps[a], colors[b], kps[b], good, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show(vis_all, "Matches %s <-> %s (%d good)" % (names[a], names[b], len(good)),
         figsize=(16, 7))

    # RANSAC inliers only.
    vis_in = cv2.drawMatches(
        colors[a], kps[a], colors[b], kps[b], good, None,
        matchColor=(0, 255, 0), matchesMask=mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show(vis_in, "RANSAC inliers %s <-> %s (%d)" % (names[a], names[b], inliers),
         figsize=(16, 7))


# %% ─────────────────────────────────────────────────────────────────────────
# CELL 6 — Manual panorama (only for 3 images, anchored on the center)
# ─────────────────────────────────────────────────────────────────────────────
pano_manual = None
if len(names) == 3:
    H_left = homographies[(0, 1)]                       # left  -> center
    H_right = np.linalg.inv(homographies[(1, 2)])       # right -> center
    pano_manual = stitch_three(colors[0], colors[1], colors[2], H_left, H_right)
    if pano_manual is not None:
        pano_manual = crop_black(pano_manual)
        show(pano_manual, "Manual panorama (SIFT + RANSAC homography)",
             figsize=(18, 8))
else:
    print("Manual 3-image stitch skipped (need exactly 3 images).")


# %% ─────────────────────────────────────────────────────────────────────────
# CELL 7 — OpenCV built-in panorama (robust comparison) + download results
# ─────────────────────────────────────────────────────────────────────────────
status, pano_cv = opencv_stitch(colors)
if status == cv2.Stitcher_OK:
    show(pano_cv, "OpenCV cv2.Stitcher panorama", figsize=(18, 8))
else:
    print("cv2.Stitcher failed with status:", status)

# Optional: save and download the panoramas.
if pano_manual is not None:
    cv2.imwrite("panorama_manual.png", pano_manual)
    files.download("panorama_manual.png")
if status == cv2.Stitcher_OK:
    cv2.imwrite("panorama_opencv_stitcher.png", pano_cv)
    files.download("panorama_opencv_stitcher.png")
