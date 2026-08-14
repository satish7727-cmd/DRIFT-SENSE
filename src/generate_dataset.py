import cv2
import numpy as np
import json
from pathlib import Path


# ============================================
# DRIFT-SENSE DATASET GENERATOR
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
TEST_DIR = PROJECT_ROOT / "data" / "test"

REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


print("========================================")
print("DRIFT-SENSE DATASET GENERATION")
print("========================================")


# --------------------------------------------
# 1. Create 100 x 100 reference image
# --------------------------------------------

reference = np.zeros((100, 100), dtype=np.uint8)

# Create a distinctive wafer-like pattern
cv2.rectangle(
    reference,
    (15, 15),
    (85, 85),
    180,
    2
)

cv2.rectangle(
    reference,
    (30, 30),
    (70, 70),
    230,
    2
)

cv2.line(
    reference,
    (20, 50),
    (80, 50),
    255,
    2
)

cv2.line(
    reference,
    (50, 20),
    (50, 80),
    255,
    2
)

cv2.circle(
    reference,
    (50, 50),
    12,
    255,
    2
)


# --------------------------------------------
# 2. Save reference
# --------------------------------------------

reference_path = REFERENCE_DIR / "reference.png"

cv2.imwrite(
    str(reference_path),
    reference
)


# --------------------------------------------
# 3. Transformation parameters
# --------------------------------------------

scale = 1.3
angle = 6.0

target_x = 600
target_y = 650


# --------------------------------------------
# 4. Resize reference
# --------------------------------------------

new_width = int(
    reference.shape[1] * scale
)

new_height = int(
    reference.shape[0] * scale
)

scaled = cv2.resize(
    reference,
    (new_width, new_height),
    interpolation=cv2.INTER_LINEAR
)


# --------------------------------------------
# 5. Rotate
# --------------------------------------------

h, w = scaled.shape[:2]

center = (
    w / 2.0,
    h / 2.0
)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    angle,
    1.0
)

rotated = cv2.warpAffine(
    scaled,
    rotation_matrix,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)


# --------------------------------------------
# 6. Create 1000 x 1000 search image
# --------------------------------------------

search_image = np.zeros(
    (1000, 1000),
    dtype=np.uint8
)


# --------------------------------------------
# 7. Add background pattern
# --------------------------------------------

for y in range(20, 1000, 40):
    search_image[y:y + 2, :] = 50

for x in range(20, 1000, 40):
    search_image[:, x:x + 2] = 50


# --------------------------------------------
# 8. Insert transformed target
# --------------------------------------------

search_image[
    target_y:target_y + h,
    target_x:target_x + w
] = rotated


# --------------------------------------------
# 9. Save search image
# --------------------------------------------

test_path = TEST_DIR / "test_001.png"

cv2.imwrite(
    str(test_path),
    search_image
)


# --------------------------------------------
# 10. Save ground truth
# --------------------------------------------

ground_truth = {
    "target_top_left_x": target_x,
    "target_top_left_y": target_y,
    "scale": scale,
    "rotation": angle,
    "target_width": w,
    "target_height": h,
    "ground_truth_center_x": target_x + w / 2,
    "ground_truth_center_y": target_y + h / 2
}


ground_truth_path = TEST_DIR / "ground_truth_001.json"

with open(
    ground_truth_path,
    "w"
) as f:
    json.dump(
        ground_truth,
        f,
        indent=4
    )


# --------------------------------------------
# 11. Print results
# --------------------------------------------

print()
print("REFERENCE")
print("----------------------------------------")
print("Path:", reference_path)
print("Shape:", reference.shape)


print()
print("TEST IMAGE")
print("----------------------------------------")
print("Path:", test_path)
print("Shape:", search_image.shape)


print()
print("GROUND TRUTH")
print("----------------------------------------")
print("Top-left:", (target_x, target_y))
print(
    "Center:",
    (
        target_x + w / 2,
        target_y + h / 2
    )
)
print("Scale:", scale)
print("Rotation:", angle)
print("Size:", (w, h))


print()
print("Saved files:")
print(reference_path)
print(test_path)
print(ground_truth_path)


print()
print("========================================")
print("DATASET GENERATION COMPLETE")
print("========================================")