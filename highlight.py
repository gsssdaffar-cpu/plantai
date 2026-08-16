import cv2
import numpy as np



def create_highlight(
        input_path,
        output_path
):


    image = cv2.imread(input_path)


    if image is None:

        return False



    # Convert to HSV

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )



    # Green leaf mask

    lower_green = np.array(
        [25,40,40]
    )


    upper_green = np.array(
        [90,255,255]
    )



    leaf_mask = cv2.inRange(

        hsv,

        lower_green,

        upper_green

    )



    # Detect non-green regions

    disease_mask = cv2.bitwise_not(
        leaf_mask
    )



    # Remove noise

    kernel = np.ones(
        (5,5),
        np.uint8
    )


    disease_mask = cv2.morphologyEx(

        disease_mask,

        cv2.MORPH_OPEN,

        kernel

    )



    # Create red overlay

    overlay = image.copy()


    overlay[disease_mask > 0] = (

        0,

        0,

        255

    )



    # Blend original + overlay

    result = cv2.addWeighted(

        image,

        0.65,

        overlay,

        0.35,

        0

    )



    # Save result

    cv2.imwrite(

        output_path,

        result

    )


    return True