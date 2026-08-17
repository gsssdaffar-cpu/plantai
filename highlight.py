# -*- coding: utf-8 -*-

"""
============================================================
PlantAI - Lightweight AI Attention Map
============================================================

Memory optimized for Render 512 MB.

This module does NOT:
    - Load another AI model
    - Use pytorch-grad-cam
    - Use torch
    - Perform Grad-CAM
    - Create the old AI Highlight

It creates an attention-style visualization using:
    - Leaf segmentation
    - Color deviation
    - Local contrast
    - Gaussian blur
    - Heatmap visualization

Output:
    *_attention.jpg

IMPORTANT:
This is an attention visualization, not true Grad-CAM.
"""


import os

import cv2
import numpy as np


# ============================================================
# CREATE ATTENTION MAP
# ============================================================

def create_attention_map(
    input_path,
    output_path
):

    image = None
    hsv = None
    leaf_mask = None
    attention = None
    heatmap = None
    result = None

    try:

        # ====================================================
        # CHECK INPUT
        # ====================================================

        if not os.path.exists(
            input_path
        ):

            print(
                "Attention input not found:",
                input_path
            )

            return False


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = cv2.imread(
            input_path,
            cv2.IMREAD_COLOR
        )


        if image is None:

            print(
                "Could not read image:",
                input_path
            )

            return False


        # ====================================================
        # RESIZE
        #
        # Important for Render memory.
        # ====================================================

        height, width = image.shape[:2]


        max_size = 768


        if max(
            height,
            width
        ) > max_size:

            scale = (
                max_size
                / float(
                    max(
                        height,
                        width
                    )
                )
            )


            new_width = max(
                1,
                int(
                    width * scale
                )
            )


            new_height = max(
                1,
                int(
                    height * scale
                )
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

            [20, 25, 25],

            dtype=np.uint8

        )


        upper_green = np.array(

            [100, 255, 255],

            dtype=np.uint8

        )


        leaf_mask = cv2.inRange(

            hsv,

            lower_green,

            upper_green

        )


        # ====================================================
        # CLEAN MASK
        # ====================================================

        kernel = np.ones(

            (5, 5),

            np.uint8

        )


        leaf_mask = cv2.morphologyEx(

            leaf_mask,

            cv2.MORPH_OPEN,

            kernel

        )


        leaf_mask = cv2.morphologyEx(

            leaf_mask,

            cv2.MORPH_CLOSE,

            kernel

        )


        # ====================================================
        # SATURATION / VALUE INFORMATION
        # ====================================================

        saturation = hsv[:, :, 1]

        value = hsv[:, :, 2]


        # ====================================================
        # COLOR DEVIATION
        #
        # Healthy green leaves generally have:
        #     moderate/high saturation
        #     green hue
        #
        # Areas with unusual color receive
        # higher attention.
        # ====================================================

        hue = hsv[:, :, 0].astype(
            np.float32
        )


        green_distance = np.abs(

            hue - 60.0

        )


        # Hue wraps around 180 in OpenCV.
        green_distance = np.minimum(

            green_distance,

            180.0 - green_distance

        )


        green_distance = cv2.normalize(

            green_distance,

            None,

            0,

            255,

            cv2.NORM_MINMAX

        )


        # ====================================================
        # LOW SATURATION / COLOR ABNORMALITY
        # ====================================================

        low_saturation = (

            255.0
            - saturation.astype(
                np.float32
            )

        )


        # ====================================================
        # DARK / BRIGHT ABNORMALITY
        # ====================================================

        brightness_difference = np.abs(

            value.astype(
                np.float32
            )
            - 140.0

        )


        brightness_difference = cv2.normalize(

            brightness_difference,

            None,

            0,

            255,

            cv2.NORM_MINMAX

        )


        # ====================================================
        # COMBINE ATTENTION FEATURES
        # ====================================================

        attention = (

            0.55 * green_distance

            +

            0.30 * low_saturation

            +

            0.15 * brightness_difference

        )


        attention = np.clip(

            attention,

            0,

            255

        ).astype(
            np.uint8
        )


        # ====================================================
        # LIMIT ATTENTION TO LEAF
        # ====================================================

        attention = cv2.bitwise_and(

            attention,

            attention,

            mask=leaf_mask

        )


        # ====================================================
        # SMOOTH ATTENTION
        # ====================================================

        attention = cv2.GaussianBlur(

            attention,

            (0, 0),

            7

        )


        # ====================================================
        # NORMALIZE
        # ====================================================

        min_value, max_value, _, _ = (
            cv2.minMaxLoc(
                attention
            )
        )


        if max_value > min_value:

            attention = cv2.normalize(

                attention,

                None,

                0,

                255,

                cv2.NORM_MINMAX

            )


        # ====================================================
        # HEATMAP
        # ====================================================

        heatmap = cv2.applyColorMap(

            attention,

            cv2.COLORMAP_JET

        )


        # ====================================================
        # ONLY SHOW HEATMAP ON LEAF
        # ====================================================

        heatmap = cv2.bitwise_and(

            heatmap,

            heatmap,

            mask=leaf_mask

        )


        # ====================================================
        # BLEND WITH ORIGINAL
        # ====================================================

        result = cv2.addWeighted(

            image,

            0.55,

            heatmap,

            0.45,

            0

        )


        # ====================================================
        # SAVE
        # ====================================================

        success = cv2.imwrite(

            output_path,

            result,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]

        )


        if not success:

            print(
                "Failed to save attention map:",
                output_path
            )

            return False


        print(
            "Attention map created:",
            output_path
        )


        return True


    except Exception as e:

        print(
            "Attention map error:",
            e
        )

        return False


    finally:

        # ====================================================
        # RELEASE NUMPY / OPENCV OBJECTS
        # ====================================================

        image = None
        hsv = None
        leaf_mask = None
        attention = None
        heatmap = None
        result = None