import cv2
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from detector import rotation_scale_detector


# ============================================
# STEP 44A - CORRECTED ROBUSTNESS TEST
# ============================================

print("========================================")
print("STEP 44A - CORRECTED ROBUSTNESS TEST")
print("========================================")


# --------------------------------------------
# PROJECT PATHS
# --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "reference.png"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------
# LOAD REFERENCE
# --------------------------------------------

reference = cv2.imread(
    str(REFERENCE_PATH),
    cv2.IMREAD_GRAYSCALE
)

if reference is None:
    raise FileNotFoundError(
        f"Reference image not found: {REFERENCE_PATH}"
    )


# --------------------------------------------
# GROUND-TRUTH POSITION
# --------------------------------------------

GT_X = 600
GT_Y = 650


# ============================================
# IMAGE TRANSFORMATION FUNCTIONS
# ============================================

def transform_reference(
    reference,
    scale,
    angle
):
    """
    Resize and rotate the reference image.
    """

    h, w = reference.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    scaled = cv2.resize(
        reference,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    h2, w2 = scaled.shape[:2]

    center = (
        w2 / 2.0,
        h2 / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        scaled,
        matrix,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated


def add_noise(
    image,
    sigma
):
    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    result = (
        image.astype(np.float32)
        +
        noise
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def change_brightness(
    image,
    factor
):
    result = (
        image.astype(np.float32)
        *
        factor
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def add_blur(
    image,
    kernel_size=5
):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


# ============================================
# CREATE TEST SCENE
# ============================================

def create_scene(
    scale=1.0,
    angle=0.0,
    noise_sigma=0,
    brightness=1.0,
    blur=False
):
    """
    Creates a 1000x1000 scene.

    The transformed target is always placed
    at the known ground-truth position
    (600,650).
    """

    target = transform_reference(
        reference,
        scale,
        angle
    )

    # ----------------------------------------
    # Apply target distortions
    # ----------------------------------------

    if brightness != 1.0:

        target = change_brightness(
            target,
            brightness
        )

    if noise_sigma > 0:

        target = add_noise(
            target,
            noise_sigma
        )

    if blur:

        target = add_blur(
            target,
            5
        )


    # ----------------------------------------
    # Create 1000 x 1000 background
    # ----------------------------------------

    scene = np.zeros(
        (1000, 1000),
        dtype=np.uint8
    )


    # ----------------------------------------
    # Add subtle background grid
    # ----------------------------------------

    for y in range(
        20,
        1000,
        40
    ):
        scene[
            y:y + 1,
            :
        ] = 35

    for x in range(
        20,
        1000,
        40
    ):
        scene[
            :,
            x:x + 1
        ] = 35


    # ----------------------------------------
    # Target dimensions
    # ----------------------------------------

    th, tw = target.shape[:2]


    # ----------------------------------------
    # Safety check
    # ----------------------------------------

    if (
        GT_Y + th > scene.shape[0]
        or
        GT_X + tw > scene.shape[1]
    ):
        raise ValueError(
            "Target does not fit inside scene."
        )


    # ----------------------------------------
    # Insert target
    # ----------------------------------------

    scene[
        GT_Y:GT_Y + th,
        GT_X:GT_X + tw
    ] = target


    return scene


# ============================================
# ROBUSTNESS TEST CASES
# ============================================

test_cases = [

    {
        "name": "Clean",
        "scale": 1.0,
        "angle": 0.0,
        "noise": 0,
        "brightness": 1.0,
        "blur": False
    },

    {
        "name": "Noise",
        "scale": 1.0,
        "angle": 0.0,
        "noise": 10,
        "brightness": 1.0,
        "blur": False
    },

    {
        "name": "Brightness",
        "scale": 1.0,
        "angle": 0.0,
        "noise": 0,
        "brightness": 1.15,
        "blur": False
    },

    {
        "name": "Blur",
        "scale": 1.0,
        "angle": 0.0,
        "noise": 0,
        "brightness": 1.0,
        "blur": True
    },

    {
        "name": "Rotation",
        "scale": 1.0,
        "angle": 6.0,
        "noise": 0,
        "brightness": 1.0,
        "blur": False
    },

    {
        "name": "Noise + Rotation",
        "scale": 1.0,
        "angle": 6.0,
        "noise": 10,
        "brightness": 1.0,
        "blur": False
    },

    {
        "name": "Blur + Rotation",
        "scale": 1.0,
        "angle": 6.0,
        "noise": 0,
        "brightness": 1.0,
        "blur": True
    },

    {
        "name": "Noise + Blur + Rotation",
        "scale": 1.0,
        "angle": 6.0,
        "noise": 10,
        "brightness": 1.0,
        "blur": True
    },

    {
        "name": "Scale 1.3x",
        "scale": 1.3,
        "angle": 0.0,
        "noise": 0,
        "brightness": 1.0,
        "blur": False
    },

    {
        "name": "Full Combined Distortion",
        "scale": 1.3,
        "angle": 6.0,
        "noise": 10,
        "brightness": 1.10,
        "blur": True
    }
]


# ============================================
# RUN TESTS
# ============================================

results = []


for i, case in enumerate(test_cases):

    print()
    print("----------------------------------------")
    print(
        f"TEST {i + 1}/{len(test_cases)}:",
        case["name"]
    )
    print("----------------------------------------")


    # ----------------------------------------
    # Create controlled test image
    # ----------------------------------------

    test_image = create_scene(
        scale=case["scale"],
        angle=case["angle"],
        noise_sigma=case["noise"],
        brightness=case["brightness"],
        blur=case["blur"]
    )


    # ----------------------------------------
    # Run detector
    # ----------------------------------------

    start = time.perf_counter()

    try:

        detection = rotation_scale_detector(
            test_image,
            reference
        )

        detection_time = (
            time.perf_counter() - start
        ) * 1000


        score = float(
            detection["score"]
        )

        detected_x, detected_y = (
            detection["location"]
        )

        detected_scale = float(
            detection["scale"]
        )

        detected_angle = float(
            detection["angle"]
        )


        # ------------------------------------
        # Errors
        # ------------------------------------

        localization_error = float(
            np.sqrt(
                (detected_x - GT_X) ** 2
                +
                (detected_y - GT_Y) ** 2
            )
        )

        scale_error = float(
            abs(
                detected_scale
                -
                case["scale"]
            )
        )

        rotation_error = float(
            abs(
                detected_angle
                -
                case["angle"]
            )
        )


        success = bool(
            score >= 0.80
            and
            localization_error <= 10
        )


    except Exception as error:

        print(
            "ERROR:",
            error
        )

        score = np.nan

        detected_x = np.nan
        detected_y = np.nan

        detected_scale = np.nan
        detected_angle = np.nan

        localization_error = np.nan

        scale_error = np.nan
        rotation_error = np.nan

        success = False

        detection_time = (
            time.perf_counter() - start
        ) * 1000


    # ----------------------------------------
    # Store result
    # ----------------------------------------

    results.append({

        "Case": i + 1,

        "Condition": case["name"],

        "Expected Scale": case["scale"],

        "Expected Angle": case["angle"],

        "Score": score,

        "Detected X": detected_x,

        "Detected Y": detected_y,

        "Localization Error": localization_error,

        "Detected Scale": detected_scale,

        "Scale Error": scale_error,

        "Detected Angle": detected_angle,

        "Rotation Error": rotation_error,

        "Detection Time (ms)": detection_time,

        "Success": success
    })


    # ----------------------------------------
    # Print result
    # ----------------------------------------

    print(
        "Score:",
        round(score, 6)
    )

    print(
        "Detected location:",
        (detected_x, detected_y)
    )

    print(
        "Expected location:",
        (GT_X, GT_Y)
    )

    print(
        "Detected scale:",
        detected_scale
    )

    print(
        "Expected scale:",
        case["scale"]
    )

    print(
        "Detected rotation:",
        detected_angle
    )

    print(
        "Expected rotation:",
        case["angle"]
    )

    print(
        "Localization error:",
        round(localization_error, 4)
    )

    print(
        "Success:",
        success
    )


# ============================================
# DATAFRAME
# ============================================

results_df = pd.DataFrame(
    results
)


# ============================================
# SAVE CSV
# ============================================

csv_path = (
    RESULTS_DIR
    / "robustness_results_corrected.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================
# SUMMARY
# ============================================

valid_scores = results_df[
    "Score"
].dropna()

valid_errors = results_df[
    "Localization Error"
].dropna()


average_score = valid_scores.mean()

minimum_score = valid_scores.min()

maximum_score = valid_scores.max()

average_error = valid_errors.mean()

maximum_error = valid_errors.max()

success_rate = (
    results_df["Success"].mean()
    *
    100
)

average_time = (
    results_df["Detection Time (ms)"].mean()
)


# ============================================
# FINAL SUMMARY
# ============================================

print()
print("========================================")
print("CORRECTED ROBUSTNESS SUMMARY")
print("========================================")

print(
    "Number of test cases:",
    len(results_df)
)

print(
    "Average matching score:",
    round(average_score, 6)
)

print(
    "Minimum matching score:",
    round(minimum_score, 6)
)

print(
    "Maximum matching score:",
    round(maximum_score, 6)
)

print(
    "Average localization error:",
    round(average_error, 4),
    "pixels"
)

print(
    "Maximum localization error:",
    round(maximum_error, 4),
    "pixels"
)

print(
    "Detection success rate:",
    round(success_rate, 2),
    "%"
)

print(
    "Average detection time:",
    round(average_time, 3),
    "ms"
)

print()
print("Results saved to:")
print(csv_path)

print()
print("========================================")
print("STEP 44A COMPLETE")
print("========================================")