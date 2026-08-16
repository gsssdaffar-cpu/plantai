import cv2
import numpy as np


def calculate_disease_area(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return 0

    image = cv2.resize(image, (512, 512))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -------- Detect the leaf --------
    lower_leaf = np.array([25, 30, 30])
    upper_leaf = np.array([95, 255, 255])

    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    kernel = np.ones((5,5), np.uint8)

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    leaf_pixels = cv2.countNonZero(leaf_mask)

    if leaf_pixels == 0:
        return 0

    # -------- Detect unhealthy colors --------
    brown1 = np.array([5, 50, 20])
    brown2 = np.array([25, 255, 255])

    yellow1 = np.array([15, 40, 60])
    yellow2 = np.array([35, 255, 255])

    brown_mask = cv2.inRange(hsv, brown1, brown2)
    yellow_mask = cv2.inRange(hsv, yellow1, yellow2)

    disease_mask = cv2.bitwise_or(
        brown_mask,
        yellow_mask
    )

    disease_mask = cv2.bitwise_and(
        disease_mask,
        leaf_mask
    )

    disease_pixels = cv2.countNonZero(disease_mask)

    percentage = (disease_pixels / leaf_pixels) * 100

    return round(percentage, 2)