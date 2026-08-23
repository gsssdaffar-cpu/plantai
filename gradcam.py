import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer(model):
    """Dynamically finds the last convolutional layer for common PyTorch models."""
    if hasattr(model, "features"):
        return [model.features[-1]]  # EfficientNet / MobileNet
    elif hasattr(model, "layer4"):
        return [model.layer4[-1]]    # ResNet / WideResNet
    else:
        # Fallback to the last layer in the sequential backbone
        return [list(model.children())[-2]]

def create_gradcam(
    model,
    image_tensor,
    image_path,
    output_path,
    target_category=None
):
    model.eval()

    # Get proper target layer automatically
    target_layers = get_target_layer(model)

    # Initialize GradCAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    try:
        # Generate Grad-CAM activation map
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=target_category
        )[0]
    finally:
        # Release PyTorch hooks to avoid GPU/CPU memory leaks
        if hasattr(cam, "activations_and_gradients"):
            cam.activations_and_gradients.release()

    # Load and process original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Resize image to match model input dimensions
    original = cv2.resize(original, (224, 224))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_normalized = original_rgb.astype(np.float32) / 255.0

    # Overlay Grad-CAM onto original image
    visualization = show_cam_on_image(
        original_normalized,
        grayscale_cam,
        use_rgb=True
    )

    # Convert back to BGR for OpenCV saving
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # Save output image
    cv2.imwrite(output_path, visualization_bgr)

    return output_path