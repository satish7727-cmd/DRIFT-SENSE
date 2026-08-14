import cv2
import numpy as np

from detector import rotation_scale_detector


# ============================================
# STEP 41E - SYNTHETIC DETECTOR TEST
# ============================================

print("========================================")
print("STEP 41E - SYNTHETIC DETECTOR TEST")
print("========================================")


# --------------------------------------------
# Create reference image
# --------------------------------------------

reference = np.zeros((100, 100), dtype=np.uint8)

cv2.rectangle(
    reference,
    (20, 20),
    (80, 80),
    255,
    -1
)

cv2.circle(
    reference,
    (50, 50),
    15,
    0,
    -1
)


# --------------------------------------------
# Create search image
# --------------------------------------------

search_image = np.zeros(
    (1000, 1000),
    dtype=np.uint8
)


# --------------------------------------------
# Scale reference to 1.3x
# --------------------------------------------

scale = 1.3

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
# Rotate by 6 degrees
# --------------------------------------------

h, w = scaled.shape[:2]

center = (
    w / 2.0,
    h / 2.0
)

matrix = cv2.getRotationMatrix2D(
    center,
    6,
    1.0
)

rotated = cv2.warpAffine(
    scaled,
    matrix,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)


# --------------------------------------------
# Insert at known position
# --------------------------------------------

ground_truth_x = 600
ground_truth_y = 650

search_image[
    ground_truth_y:ground_truth_y + h,
    ground_truth_x:ground_truth_x + w
] = rotated


# --------------------------------------------
# Run detector
# --------------------------------------------

print()
print("Running detector...")
print("Ground-truth position:",
      (ground_truth_x, ground_truth_y))

print("Ground-truth scale:",
      scale)

print("Ground-truth rotation:",
      6,
      "degrees")

print()


result = rotation_scale_detector(
    search_image,
    reference
)


# --------------------------------------------
# Display result
# --------------------------------------------

print("========================================")
print("DETECTOR RESULT")
print("========================================")

print(
    "Matching score:",
    result["score"]
)

print(
    "Detected location:",
    result["location"]
)

print(
    "Detected scale:",
    result["scale"]
)

print(
    "Detected rotation:",
    result["angle"]
)

print(
    "Detected size:",
    result["size"]
)


# --------------------------------------------
# Calculate localization error
# --------------------------------------------

detected_x, detected_y = result["location"]

localization_error = np.sqrt(
    (detected_x - ground_truth_x) ** 2
    +
    (detected_y - ground_truth_y) ** 2
)

print(
    "Localization error:",
    localization_error,
    "pixels"
)


# --------------------------------------------
# Final status
# --------------------------------------------

success = (
    result["score"] >= 0.80
    and
    localization_error <= 10
)

print()
print("Success:", success)

print("========================================")
print("STEP 41E COMPLETE")
print("========================================")