# ============================================================
# Lowpass Filtering & Thresholding for Region Extraction
# Hubble Telescope - Hickson Compact Group
# ============================================================

# --------------- CELL 1: Install & Import ---------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import io, img_as_float, filters
from skimage.color import rgb2gray
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# --------------- CELL 2: Upload or Download Image ---------------
# Option A: Upload your own image
uploaded = files.upload()
filename = list(uploaded.keys())[0]
image_raw = io.imread(filename)
print(f"Image loaded: {filename}, shape: {image_raw.shape}")

# --------------- CELL 3: Preprocess - Convert to Grayscale ---------------
# Convert to float grayscale [0, 1]
if image_raw.ndim == 3:
    image_gray = rgb2gray(img_as_float(image_raw))
else:
    image_gray = img_as_float(image_raw)

print(f"Grayscale image shape: {image_gray.shape}")
print(f"Intensity range: [{image_gray.min():.4f}, {image_gray.max():.4f}]")

# --------------- CELL 4: Apply Gaussian Lowpass Filter ---------------
# Gaussian kernel sigma controls blur strength
# Larger sigma = more smoothing (removes more high-frequency noise)
sigma = 3  # Try values: 1, 2, 3, 5, 10

image_filtered = gaussian_filter(image_gray, sigma=sigma)

print(f"Gaussian filter applied with sigma={sigma}")
print(f"Filtered intensity range: [{image_filtered.min():.4f}, {image_filtered.max():.4f}]")

# --------------- CELL 5: Thresholding for Region Extraction ---------------
# Scale intensities to [0, 1] then apply threshold
image_normalized = (image_filtered - image_filtered.min()) / (image_filtered.max() - image_filtered.min())

# Method 1: Manual threshold
threshold_value = 0.15  # Adjust this value (0.0 - 1.0)

# Method 2: Otsu's automatic threshold (uncomment to use)
# threshold_value = filters.threshold_otsu(image_normalized)
# print(f"Otsu threshold: {threshold_value:.4f}")

image_thresholded = image_normalized > threshold_value

print(f"Threshold value: {threshold_value}")
print(f"Pixels above threshold: {image_thresholded.sum()} ({100*image_thresholded.mean():.2f}%)")

# --------------- CELL 6: Visualize Results (a), (b), (c) ---------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Lowpass Filtering & Thresholding - Hickson Compact Group\n(Hubble Space Telescope)',
             fontsize=14, fontweight='bold', y=1.02)

# (a) Original image
axes[0].imshow(image_gray, cmap='gray', aspect='auto')
axes[0].set_title(f'(a) Original Image\n{image_gray.shape[1]}×{image_gray.shape[0]} pixels',
                  fontsize=12, color='navy')
axes[0].axis('off')

# (b) Lowpass filtered (Gaussian)
axes[1].imshow(image_filtered, cmap='gray', aspect='auto')
axes[1].set_title(f'(b) Gaussian Lowpass Filter\n(sigma={sigma})',
                  fontsize=12, color='navy')
axes[1].axis('off')

# (c) Thresholded binary image
axes[2].imshow(image_thresholded, cmap='gray', aspect='auto')
axes[2].set_title(f'(c) Thresholded Image\n(threshold={threshold_value}, scaled to [0,1])',
                  fontsize=12, color='navy')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('result_lowpass_thresholding.png', dpi=150, bbox_inches='tight')
plt.show()
print("Result saved as 'result_lowpass_thresholding.png'")

# --------------- CELL 7: Experiment with Different Sigma Values ---------------
sigmas = [1, 3, 5, 10, 20]

fig, axes = plt.subplots(2, len(sigmas), figsize=(20, 8))
fig.suptitle('Effect of Different Gaussian Sigma Values', fontsize=14, fontweight='bold')

for i, s in enumerate(sigmas):
    filtered = gaussian_filter(image_gray, sigma=s)
    normalized = (filtered - filtered.min()) / (filtered.max() - filtered.min())
    thresholded = normalized > threshold_value

    axes[0, i].imshow(filtered, cmap='gray')
    axes[0, i].set_title(f'Filtered\nsigma={s}', fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].imshow(thresholded, cmap='gray')
    axes[1, i].set_title(f'Thresholded\nsigma={s}', fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('result_sigma_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

# --------------- CELL 8: Experiment with Different Threshold Values ---------------
thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
best_filtered = gaussian_filter(image_gray, sigma=sigma)
best_normalized = (best_filtered - best_filtered.min()) / (best_filtered.max() - best_filtered.min())

fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 4))
fig.suptitle(f'Effect of Different Threshold Values (sigma={sigma})', fontsize=14, fontweight='bold')

for i, t in enumerate(thresholds):
    binary = best_normalized > t
    axes[i].imshow(binary, cmap='gray')
    axes[i].set_title(f'Threshold={t}\n{binary.sum()} pixels', fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('result_threshold_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

# --------------- CELL 9: Download Results ---------------
files.download('result_lowpass_thresholding.png')
files.download('result_sigma_comparison.png')
files.download('result_threshold_comparison.png')
print("Done! Results downloaded.")
