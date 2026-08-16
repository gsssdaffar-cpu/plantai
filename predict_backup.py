import os

import torch
from PIL import Image
from torchvision import transforms

from config import DEVICE, IMAGE_SIZE, MODEL_PATH
from model import create_model


# ============================================================
# FIND THE 38 CLASSES
# ============================================================

DATASET_PATH = "dataset_global"


def get_class_names():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    classes = [
        name
        for name in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, name)
        )
    ]

    classes.sort()

    if len(classes) == 0:

        raise RuntimeError(
            "No classes found in dataset_global."
        )

    return classes


CLASS_NAMES = get_class_names()


print(
    f"Prediction classes loaded: {len(CLASS_NAMES)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = create_model(
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


print(
    f"Loaded model: {MODEL_PATH}"
)


# ============================================================
# IMAGE TRANSFORM
#
# IMPORTANT:
# The model was trained using Resize + ToTensor,
# so inference uses the same preprocessing.
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()

])


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)


    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )


    class_index = predicted.item()

    confidence_value = (
        confidence.item() * 100
    )


    prediction = CLASS_NAMES[
        class_index
    ]


    return (
        prediction,
        confidence_value
    )


# ============================================================
# MODEL ACCESS
#
# Used by Grad-CAM
# ============================================================

def get_model():

    return model


# ============================================================
# TRANSFORM ACCESS
# ============================================================

def get_transform():

    return transform


# ============================================================
# CLASS ACCESS
# ============================================================

def get_classes():

    return CLASS_NAMES