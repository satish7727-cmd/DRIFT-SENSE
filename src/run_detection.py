import cv2
import json
import numpy as np

from pathlib import Path
from detector import rotation_scale_detector


# ============================================
# STEP 42F - REAL PROJECT IMAGE DETECTION
# ============================================

print("========================================")
print("STEP 42F - PROJECT IMAGE DETECTION")
print("========================================")


# --------------------------------------------
# Project paths
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

reference_path = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "reference.png"
)

test_path = (
    PROJECT_ROOT
    / "data"
    / "test"
    / "test_001.png"
)

ground_truth_path = (
    PROJECT_ROOT
    / "data"
    / "test"
    / "ground_truth_001.json"
)


# --------------------------------------------
# Load images
# --------------------------------------------

reference = cv2.imread(
    str(reference_path),
    cv2.IMREAD_GRAYSCALE
)

test_image = cv2.imread(
    str(test_path),
    cv2.IMREAD_GRAYSCALE
)


if reference is None:
    raise FileNotFoundError(
        f"Reference image not found: {reference_path}"
    )

if test_image is None:
    raise FileNotFoundError(
        f"Test image not found: {test_path}"
    )


# --------------------------------------------
# Load ground truth
# --------------------------------------------

with open(
    ground_truth_path,
    "r"
) as f:

    ground_truth = json.load(f)


gt_x = ground_truth["target_top_left_x"]
gt_y = ground_truth["target_top_left_y"]

gt_scale = ground_truth["scale"]
gt_angle = ground_truth["rotation"]


# --------------------------------------------
# Print input information
# --------------------------------------------

print()
print("REFERENCE")
print("----------------------------------------")
print("Shape:", reference.shape)
print("Path:", reference_path)


print()
print("TEST IMAGE")
print("----------------------------------------")
print("Shape:", test_image.shape)
print("Path:", test_path)


print()
print("GROUND TRUTH")
print("----------------------------------------")
print("Top-left:", (gt_x, gt_y))
print("Scale:", gt_scale)
print("Rotation:", gt_angle)


# --------------------------------------------
# Run detector
# --------------------------------------------

print()
print("Running multi-scale + rotation detector...")
print("Please wait...")


result = rotation_scale_detector(
    test_image,
    reference
)


# --------------------------------------------
# Extract result
# --------------------------------------------

score = result["score"]

detected_x, detected_y = result["location"]

detected_scale = result["scale"]

detected_angle = result["angle"]

detected_width, detected_height = result["size"]


# --------------------------------------------
# Localization error
# --------------------------------------------

localization_error = np.sqrt(
    (detected_x - gt_x) ** 2
    +
    (detected_y - gt_y) ** 2
)


# --------------------------------------------
# Parameter errors
# --------------------------------------------

scale_error = abs(
    detected_scale - gt_scale
)

rotation_error = abs(
    detected_angle - gt_angle
)


# --------------------------------------------
# Success criterion
# --------------------------------------------

success = (
    score >= 0.80
    and
    localization_error <= 10
)


# --------------------------------------------
# Print results
# --------------------------------------------

print()
print("========================================")
print("DETECTION RESULT")
print("========================================")

print(
    "Matching score:",
    round(score, 6)
)

print(
    "Detected top-left:",
    (detected_x, detected_y)
)

print(
    "Detected size:",
    (detected_width, detected_height)
)

print(
    "Detected scale:",
    detected_scale
)

print(
    "Detected rotation:",
    detected_angle,
    "degrees"
)

print(
    "Localization error:",
    round(localization_error, 4),
    "pixels"
)

print(
    "Scale error:",
    scale_error
)

print(
    "Rotation error:",
    rotation_error,
    "degrees"
)

print(
    "Success:",
    success
)

print()
print("========================================")
print("STEP 42F COMPLETE")
print("========================================")