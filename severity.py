import cv2
import numpy as np


def estimate_severity(image_path):

    image = cv2.imread(image_path)


    if image is None:

        return {
            "level": "Unknown",
            "area": 0
        }



    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # Detect yellow/brown infected areas

    lower = np.array(
        [10, 30, 30]
    )

    upper = np.array(
        [45, 255, 255]
    )


    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )


    infected_pixels = np.sum(mask > 0)


    total_pixels = (
        image.shape[0] *
        image.shape[1]
    )


    affected_area = (
        infected_pixels /
        total_pixels
    ) * 100



    if affected_area < 20:

        level = "Mild"


    elif affected_area < 50:

        level = "Moderate"


    else:

        level = "Critical"



    return {

        "level": level,

        "area": round(
            affected_area,
            2
        )

    }