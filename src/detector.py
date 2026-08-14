import cv2
import numpy as np


# ============================================
# ROTATION HELPER
# ============================================

def rotate_image_keep_size(image, angle):

    h, w = image.shape[:2]

    center = (
        w / 2.0,
        h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated


# ============================================
# MULTI-SCALE + ROTATION DETECTOR
# ============================================

def rotation_scale_detector(
    search_image,
    template,
    angles=None,
    scales=None
):

    if angles is None:

        angles = np.arange(
            -10,
            11,
            2
        )

    if scales is None:

        scales = np.linspace(
            0.5,
            1.5,
            21
        )

    best_score = -1.0
    best_location = None
    best_angle = None
    best_scale = None
    best_size = None

    for angle in angles:

        rotated_template = (
            rotate_image_keep_size(
                template,
                angle
            )
        )

        for scale in scales:

            width = int(
                rotated_template.shape[1]
                * scale
            )

            height = int(
                rotated_template.shape[0]
                * scale
            )

            if width < 5 or height < 5:
                continue

            if (
                width > search_image.shape[1]
                or
                height > search_image.shape[0]
            ):
                continue

            scaled_template = cv2.resize(
                rotated_template,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

            result = cv2.matchTemplate(
                search_image,
                scaled_template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_value, _, max_location = (
                cv2.minMaxLoc(result)
            )

            if max_value > best_score:

                best_score = float(
                    max_value
                )

                best_location = (
                    max_location
                )

                best_angle = float(
                    angle
                )

                best_scale = float(
                    scale
                )

                best_size = (
                    width,
                    height
                )

    return {
        "score": best_score,
        "location": best_location,
        "angle": best_angle,
        "scale": best_scale,
        "size": best_size
    }