# -*- coding: utf-8 -*-

from pathlib import Path
import threading

import torch
import numpy as np

from PIL import Image

from predict import (
    get_model,
    get_transform,
    get_class_names,
    DEVICE
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCK
#
# Only one Grad-CAM operation at a time.
# This helps prevent memory spikes on Render.
# ============================================================

GRADCAM_LOCK = threading.Lock()


# ============================================================
# GENERATE ATTENTION MAP
# ============================================================

def generate_attention_map(
    image_path,
    output_path,
    class_index=None
):

    """
    Generate a lightweight Grad-CAM attention map
    for EfficientNet-B0.

    Parameters
    ----------
    image_path:
        Original uploaded image.

    output_path:
        Where attention map should be saved.

    class_index:
        Target class index.
        If None, the model's predicted class is used.

    Returns
    -------
    output_path
        Path of generated attention map.
    """


    # ========================================================
    # PREVENT MULTIPLE GRAD-CAM JOBS
    # ========================================================

    acquired = GRADCAM_LOCK.acquire(
        timeout=5
    )

    if not acquired:

        raise RuntimeError(
            "Another attention-map request is already running."
        )


    # ========================================================
    # VARIABLES
    # ========================================================

    model = get_model()
    transform = get_transform()
    class_names = get_class_names()

    activations = None
    gradients = None


    # ========================================================
    # HOOK FUNCTIONS
    # ========================================================

    def forward_hook(
        module,
        input_data,
        output
    ):

        nonlocal activations

        activations = output


    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradients

        gradients = grad_output[0]


    forward_handle = None
    backward_handle = None


    try:

        # ====================================================
        # TARGET LAYER
        #
        # EfficientNet-B0:
        #
        # model.features[-1]
        #
        # This is the final convolutional feature layer.
        # ====================================================

        target_layer = model.features[-1]


        forward_handle = target_layer.register_forward_hook(
            forward_hook
        )


        backward_handle = target_layer.register_full_backward_hook(
            backward_hook
        )


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # ====================================================
        # TRANSFORM
        # ====================================================

        image_tensor = transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )


        # ====================================================
        # ENABLE GRADIENT
        # ====================================================

        model.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # FORWARD PASS
        # ====================================================

        output = model(
            image_tensor
        )


        # ====================================================
        # TARGET CLASS
        # ====================================================

        if class_index is None:

            class_index = int(
                torch.argmax(
                    output,
                    dim=1
                ).item()
            )


        # ====================================================
        # TARGET SCORE
        # ====================================================

        target_score = output[
            0,
            class_index
        ]


        # ====================================================
        # BACKWARD PASS
        # ====================================================

        target_score.backward()


        # ====================================================
        # VALIDATE HOOK DATA
        # ====================================================

        if activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )


        if gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )


        # ====================================================
        # REMOVE BATCH DIMENSION
        # ====================================================

        activation = activations[
            0
        ]


        gradient = gradients[
            0
        ]


        # ====================================================
        # GLOBAL AVERAGE POOLING OF GRADIENTS
        # ====================================================

        weights = gradient.mean(
            dim=(1, 2)
        )


        # ====================================================
        # WEIGHT ACTIVATION MAPS
        # ====================================================

        cam = (
            weights[:, None, None]
            * activation
        ).sum(
            dim=0
        )


        # ====================================================
        # RELU
        # ====================================================

        cam = torch.relu(
            cam
        )


        # ====================================================
        # NORMALIZE
        # ====================================================

        cam_min = cam.min()
        cam_max = cam.max()


        if (
            cam_max - cam_min
        ) > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(
                cam
            )


        # ====================================================
        # MOVE TO CPU
        # ====================================================

        cam = cam.detach().cpu().numpy()


        # ====================================================
        # CONVERT TO 0-255
        # ====================================================

        heatmap = (
            cam * 255
        ).astype(
            np.uint8
        )


        # ====================================================
        # CREATE PIL IMAGE
        # ====================================================

        heatmap_image = Image.fromarray(
            heatmap
        ).convert(
            "RGB"
        )


        # ====================================================
        # RESIZE TO ORIGINAL IMAGE SIZE
        # ====================================================

        heatmap_image = heatmap_image.resize(
            image.size,
            Image.Resampling.BILINEAR
        )


        # ====================================================
        # CREATE SIMPLE RED/YELLOW ATTENTION MAP
        #
        # No OpenCV required.
        # ====================================================

        heatmap_array = np.asarray(
            heatmap_image
        ).astype(
            np.float32
        ) / 255.0


        # ====================================================
        # CREATE RGB HEATMAP
        # ====================================================

        red = heatmap_array[:, :, 0]

        heatmap_rgb = np.zeros(
            (
                heatmap_array.shape[0],
                heatmap_array.shape[1],
                3
            ),
            dtype=np.uint8
        )


        # Red channel

        heatmap_rgb[:, :, 0] = (
            red * 255
        ).astype(
            np.uint8
        )


        # Green channel

        heatmap_rgb[:, :, 1] = (
            np.clip(
                red * 2.0 - 0.5,
                0,
                1
            ) * 255
        ).astype(
            np.uint8
        )


        # Blue channel

        heatmap_rgb[:, :, 2] = (
            np.clip(
                1.0 - red * 2.0,
                0,
                1
            ) * 255
        ).astype(
            np.uint8
        )


        heatmap_image = Image.fromarray(
            heatmap_rgb
        )


        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        original = image.convert(
            "RGBA"
        )

        heatmap_rgba = heatmap_image.convert(
            "RGBA"
        )


        # 45% heatmap opacity

        alpha = heatmap_rgba.getchannel(
            "A"
        ).point(
            lambda value: int(
                value * 0.45
            )
        )


        heatmap_rgba.putalpha(
            alpha
        )


        overlay = Image.alpha_composite(
            original,
            heatmap_rgba
        )


        # ====================================================
        # SAVE
        # ====================================================

        output_path = Path(
            output_path
        )


        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        overlay.convert(
            "RGB"
        ).save(
            output_path,
            "JPEG",
            quality=85,
            optimize=True
        )


        # ====================================================
        # CLEAN TENSORS
        # ====================================================

        del image_tensor
        del output
        del target_score
        del activation
        del gradient
        del weights
        del cam


        # ====================================================
        # CUDA CLEANUP
        # ====================================================

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()


        print(
            "AI attention map generated:",
            output_path
        )


        print(
            "Attention class:",
            class_names[class_index]
        )


        return str(
            output_path
        )


    finally:

        # ====================================================
        # REMOVE HOOKS
        # ====================================================

        if forward_handle is not None:

            forward_handle.remove()


        if backward_handle is not None:

            backward_handle.remove()


        # ====================================================
        # CLEAN MODEL GRADIENTS
        # ====================================================

        model.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # RELEASE LOCK
        # ====================================================

        GRADCAM_LOCK.release()