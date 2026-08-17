# -*- coding: utf-8 -*-

from pathlib import Path
import time

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# DEVICE
# ============================================================

# Render normally runs CPU.
# Force CPU unless CUDA is genuinely available.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# CPU THREAD LIMIT
# ============================================================

# Prevent PyTorch from creating too many CPU threads
# on a small Render instance.

if DEVICE.type == "cpu":

    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print("PyTorch CPU threads:", torch.get_num_threads())


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "plant_model_38class.pth"
)

CLASS_NAMES_PATH = (
    BASE_DIR
    / "models"
    / "class_names.pth"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel file not found:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 60)
print("Loading PlantAI model...")
print("=" * 60)

load_start = time.perf_counter()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    f"Checkpoint loaded in "
    f"{time.perf_counter() - load_start:.2f} seconds"
)


# ============================================================
# GET MODEL STATE
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# GET CLASS NAMES
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "classes" in checkpoint
):

    CLASS_NAMES = checkpoint[
        "classes"
    ]

    print(
        "Classes loaded from model checkpoint."
    )

elif CLASS_NAMES_PATH.exists():

    CLASS_NAMES = torch.load(
        CLASS_NAMES_PATH,
        map_location="cpu",
        weights_only=False
    )

    print(
        "Classes loaded from class_names.pth."
    )

else:

    raise FileNotFoundError(
        "\nClass names not found:\n"
        f"{CLASS_NAMES_PATH}"
    )


# ============================================================
# VALIDATE CLASS COUNT
# ============================================================

NUM_CLASSES = len(
    CLASS_NAMES
)

print(
    "Prediction classes:",
    NUM_CLASSES
)

if NUM_CLASSES != 38:

    raise RuntimeError(
        f"Expected 38 classes, "
        f"but found {NUM_CLASSES}"
    )


# ============================================================
# CREATE EFFICIENTNET-B0
# ============================================================

def create_model():

    model = models.efficientnet_b0(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = torch.nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

model = create_model()


# ============================================================
# CLEAN STATE DICT
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]

    if key.startswith("model."):

        key = key[
            len("model.") :
        ]

    clean_state_dict[key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

print("Loading model weights...")

model.load_state_dict(
    clean_state_dict,
    strict=True
)


# ============================================================
# MOVE MODEL
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# OPTIONAL MODEL INFORMATION
# ============================================================

if isinstance(checkpoint, dict):

    if "best_accuracy" in checkpoint:

        try:

            print(
                f"Best validation accuracy: "
                f"{checkpoint['best_accuracy']:.2f}%"
            )

        except Exception:

            pass

    if "num_classes" in checkpoint:

        print(
            "Model classes:",
            checkpoint["num_classes"]
        )


# ============================================================
# CLEANUP CHECKPOINT
# ============================================================

# We don't need the complete checkpoint anymore.
# Keeping it in memory can unnecessarily increase RAM usage.

del checkpoint
del state_dict
del clean_state_dict

if DEVICE.type == "cuda":

    torch.cuda.empty_cache()


# ============================================================
# MODEL READY
# ============================================================

print()
print("✅ PlantAI prediction model loaded")
print("Model:", MODEL_PATH)
print("Device:", DEVICE)
print("=" * 60)
print()


# ============================================================
# GET MODEL
# ============================================================

def get_model():

    return model


# ============================================================
# GET TRANSFORM
# ============================================================

def get_transform():

    return transform


# ============================================================
# GET CLASS NAMES
# ============================================================

def get_class_names():

    return CLASS_NAMES


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    """
    Predict plant disease.

    Returns:

        prediction
        confidence
        top_predictions
        confidence_status
    """

    prediction_start = time.perf_counter()


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image_start = time.perf_counter()

    with Image.open(image_path) as img:

        image = img.convert("RGB")


    print(
        f"Image loaded in "
        f"{time.perf_counter() - image_start:.3f}s"
    )


    # ========================================================
    # TRANSFORM
    # ========================================================

    transform_start = time.perf_counter()

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )


    print(
        f"Image transformed in "
        f"{time.perf_counter() - transform_start:.3f}s"
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    inference_start = time.perf_counter()

    with torch.inference_mode():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(
                5,
                len(CLASS_NAMES)
            ),
            dim=1
        )


    inference_time = (
        time.perf_counter()
        - inference_start
    )


    print(
        f"Model inference time: "
        f"{inference_time:.3f}s"
    )


    # ========================================================
    # TOP 5
    # ========================================================

    top_predictions = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        index = index.item()

        confidence_value = (
            probability.item()
            * 100
        )

        class_name = CLASS_NAMES[
            index
        ]

        top_predictions.append({

            "class_name":
                class_name,

            "confidence":
                round(
                    confidence_value,
                    2
                ),

            "index":
                index

        })


    # ========================================================
    # BEST PREDICTION
    # ========================================================

    prediction = (
        top_predictions[0]["class_name"]
    )

    confidence = (
        top_predictions[0]["confidence"]
    )


    # ========================================================
    # CONFIDENCE STATUS
    # ========================================================

    if confidence < 40:

        confidence_status = "Unknown"

    elif confidence < 60:

        confidence_status = "Low confidence"

    elif confidence < 80:

        confidence_status = "Possible diagnosis"

    elif confidence < 95:

        confidence_status = "Likely diagnosis"

    else:

        confidence_status = "High confidence"


    # ========================================================
    # CLEAN TENSORS
    # ========================================================

    del image_tensor
    del output
    del probabilities
    del top_probabilities
    del top_indices

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()


    # ========================================================
    # TIMING
    # ========================================================

    total_time = (
        time.perf_counter()
        - prediction_start
    )


    # ========================================================
    # RESULT LOG
    # ========================================================

    print()
    print("=" * 60)
    print("🌱 PLANTAI PREDICTION")
    print("=" * 60)

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    print(
        "Status:",
        confidence_status
    )

    print(
        f"Inference: "
        f"{inference_time:.3f}s"
    )

    print(
        f"Total prediction time: "
        f"{total_time:.3f}s"
    )

    print()
    print("Top 5 predictions:")

    for i, item in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{i}. "
            f"{item['class_name']} "
            f"({item['confidence']:.2f}%)"
        )

    print("=" * 60)
    print()


    # ========================================================
    # RETURN
    # ========================================================

    return (

        prediction,

        confidence,

        top_predictions,

        confidence_status

    )