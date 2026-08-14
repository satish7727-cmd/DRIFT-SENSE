import cv2
import json
import numpy as np
from pathlib import Path

from detector import rotation_scale_detector


# ============================================
# STEP 43 - SAVE DETECTION RESULT
# ============================================

print("========================================")
print("STEP 43 - SAVE DETECTION RESULT")
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

results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = results_dir / "detection_result.json"


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
        f"Reference image not found:\n{reference_path}"
    )

if test_image is None:
    raise FileNotFoundError(
        f"Test image not found:\n{test_path}"
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
# Run detector
# --------------------------------------------

print()
print("Running detector...")
print("Please wait...")


result = rotation_scale_detector(
    test_image,
    reference
)


# --------------------------------------------
# Extract detection result
# --------------------------------------------

score = float(result["score"])

detected_x, detected_y = result["location"]

detected_scale = float(result["scale"])

detected_angle = float(result["angle"])

detected_width, detected_height = result["size"]


# --------------------------------------------
# Calculate localization error
# --------------------------------------------

localization_error = float(
    np.sqrt(
        (detected_x - gt_x) ** 2
        +
        (detected_y - gt_y) ** 2
    )
)


# --------------------------------------------
# Calculate parameter errors
# --------------------------------------------

scale_error = float(
    abs(detected_scale - gt_scale)
)

rotation_error = float(
    abs(detected_angle - gt_angle)
)


# --------------------------------------------
# Success condition
# --------------------------------------------

success = bool(
    score >= 0.80
    and
    localization_error <= 10
)


# --------------------------------------------
# Create final result
# --------------------------------------------

final_result = {

    "method": "Multi-scale + Rotation",

    "reference_image": str(
        reference_path.relative_to(PROJECT_ROOT)
    ),

    "test_image": str(
        test_path.relative_to(PROJECT_ROOT)
    ),

    "ground_truth": {
        "location": [
            gt_x,
            gt_y
        ],
        "scale": gt_scale,
        "rotation": gt_angle
    },

    "detection": {
        "matching_score": score,
        "location": [
            int(detected_x),
            int(detected_y)
        ],
        "size": [
            int(detected_width),
            int(detected_height)
        ],
        "scale": detected_scale,
        "rotation": detected_angle
    },

    "errors": {
        "localization_error_pixels": localization_error,
        "scale_error": scale_error,
        "rotation_error_degrees": rotation_error
    },

    "success": success
}


# --------------------------------------------
# Save JSON
# --------------------------------------------

with open(
    output_path,
    "w"
) as f:

    json.dump(
        final_result,
        f,
        indent=4
    )


# --------------------------------------------
# Print result
# --------------------------------------------

print()
print("========================================")
print("FINAL DETECTION")
print("========================================")

print(
    "Matching score:",
    round(score, 6)
)

print(
    "Detected location:",
    (detected_x, detected_y)
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
print("Result saved to:")
print(output_path)

print()
print("========================================")
print("STEP 43 COMPLETE")
print("========================================")