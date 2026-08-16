import torch
import cv2
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image



def create_gradcam(
        model,
        image_tensor,
        image_path,
        output_path
):

    model.eval()


    target_layers = [

        model.features[-1]

    ]


    cam = GradCAM(

        model=model,

        target_layers=target_layers

    )


    grayscale_cam = cam(

        input_tensor=image_tensor

    )[0]



    original = cv2.imread(
        image_path
    )


    original = cv2.cvtColor(

        original,

        cv2.COLOR_BGR2RGB

    )


    original = cv2.resize(

        original,

        (224,224)

    )


    original = original.astype(
        np.float32
    ) / 255



    visualization = show_cam_on_image(

        original,

        grayscale_cam,

        use_rgb=True

    )


    visualization = cv2.cvtColor(

        visualization,

        cv2.COLOR_RGB2BGR

    )



    cv2.imwrite(

        output_path,

        visualization

    )


    return output_path