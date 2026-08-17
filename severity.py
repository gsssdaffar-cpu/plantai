# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Severity Estimation
============================================================

Lightweight OpenCV-based severity estimation.

Returns:
    severity
    affected_area

No AI model is loaded.
No PyTorch is used.
============================================================
"""

import cv2
import numpy as np


def estimate_severity(image_path, prediction=None):

    image = None
    hsv = None
    leaf_mask = None
    infected_mask = None

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        image = cv2.imread(
            image_path
        )

        if image is None:

            return {
                "severity": "Unknown",
                "severity_level": "Unknown",
                "level": "Unknown",
                "affected_area": 0,
                "area": 0
            }


        # ====================================================
        # RESIZE
        # ====================================================

        height, width = image.shape[:2]

        max_dimension = 700

        if max(height, width) > max_dimension:

            scale = (
                max_dimension /
                float(max(height, width))
            )

            new_width = max(
                1,
                int(width * scale)
            )

            new_height = max(
                1,
                int(height * scale)
            )

            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height
                ),
                interpolation=cv2.INTER_AREA
            )


        # ====================================================
        # HSV
        # ====================================================

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )


        # ====================================================
        # LEAF MASK
        # ====================================================

        lower_green = np.array(
            [25, 30, 30],
            dtype=np.uint8
        )

        upper_green = np.array(
            [100, 255, 255],
            dtype=np.uint8
        )

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )


        # ====================================================
        # ALSO INCLUDE YELLOW/BROWN LEAF REGIONS
        # ====================================================

        lower_leaf_problem = np.array(
            [5, 25, 20],
            dtype=np.uint8
        )

        upper_leaf_problem = np.array(
            [45, 255, 255],
            dtype=np.uint8
        )

        problem_leaf_mask = cv2.inRange(
            hsv,
            lower_leaf_problem,
            upper_leaf_problem
        )


        # Combine green + yellow/brown regions
        leaf_mask = cv2.bitwise_or(
            green_mask,
            problem_leaf_mask
        )


        # ====================================================
        # CLEAN LEAF MASK
        # ====================================================

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        leaf_mask = cv2.morphologyEx(
            leaf_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        leaf_mask = cv2.morphologyEx(
            leaf_mask,
            cv2.MORPH_OPEN,
            kernel
        )


        # ====================================================
        # INFECTED REGION
        # ====================================================

        lower_infected = np.array(
            [8, 35, 20],
            dtype=np.uint8
        )

        upper_infected = np.array(
            [45, 255, 245],
            dtype=np.uint8
        )

        infected_mask = cv2.inRange(
            hsv,
            lower_infected,
            upper_infected
        )


        # ====================================================
        # ONLY COUNT INFECTION INSIDE LEAF
        # ====================================================

        infected_mask = cv2.bitwise_and(
            infected_mask,
            leaf_mask
        )


        # ====================================================
        # REMOVE SMALL NOISE
        # ====================================================

        infected_mask = cv2.morphologyEx(
            infected_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        infected_mask = cv2.morphologyEx(
            infected_mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # ====================================================
        # PIXELS
        # ====================================================

        leaf_pixels = np.sum(
            leaf_mask > 0
        )

        infected_pixels = np.sum(
            infected_mask > 0
        )


        # ====================================================
        # NO LEAF DETECTED
        # ====================================================

        if leaf_pixels <= 0:

            return {
                "severity": "Unknown",
                "severity_level": "Unknown",
                "level": "Unknown",
                "affected_area": 0,
                "area": 0
            }


        # ====================================================
        # AFFECTED AREA
        # ====================================================

        affected_area = (

            infected_pixels /

            float(leaf_pixels)

        ) * 100.0


        # ====================================================
        # LIMIT
        # ====================================================

        affected_area = max(
            0,
            min(
                100,
                affected_area
            )
        )


        affected_area = round(
            affected_area,
            2
        )


        # ====================================================
        # SEVERITY
        # ====================================================

        if affected_area < 20:

            level = "Mild"

        elif affected_area < 50:

            level = "Moderate"

        else:

            level = "Critical"


        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "severity":
                level,

            "severity_level":
                level,

            "level":
                level,

            "affected_area":
                affected_area,

            "area":
                affected_area

        }


        print(
            "Severity:",
            level
        )

        print(
            "Affected leaf area:",
            f"{affected_area:.2f}%"
        )


        return result


    except Exception as e:

        print(
            "Severity estimation error:",
            e
        )


        return {

            "severity":
                "Unknown",

            "severity_level":
                "Unknown",

            "level":
                "Unknown",

            "affected_area":
                0,

            "area":
                0

        }


    finally:

        image = None
        hsv = None
        leaf_mask = None
        infected_mask = None

        try:
            del green_mask
        except Exception:
            pass

        try:
            del problem_leaf_mask
        except Exception:
            pass

        try:
            del infected_mask
        except Exception:
            pass

        try:
            del kernel
        except Exception:
            pass

        import gc

        gc.collect()