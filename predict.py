# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Prediction Engine
============================================================

Optimized for low-memory Render deployment.

Model:
    EfficientNet-B0

Classes:
    38 PlantVillage classes

Features:
    - CPU inference
    - Lazy model loading
    - Top-5 predictions
    - Confidence status
    - Memory cleanup after every prediction

NOT INCLUDED:
    - Grad-CAM
    - Attention map
    - Severity
    - Highlight
    - PDF
    - Explanation

Those features should be handled separately.
============================================================
"""

import gc
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# DEVICE
# ============================================================

# Render deployment should use CPU.
# Explicit CPU prevents unnecessary CUDA initialization.

DEVICE = torch.device("cpu")

print("=" * 60)
print("PlantAI Prediction Engine")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# PYTORCH THREAD CONTROL
# ============================================================

# Render Free instances have limited CPU/RAM.
# Limiting threads helps reduce memory usage.

try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


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
# VERIFY MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(

        "\nPlantAI model file not found.\n\n"

        f"Expected:\n"
        f"{MODEL_PATH}\n"

    )


# ============================================================
# VERIFY CLASS NAMES
# ============================================================

if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(

        "\nPlantAI class names file not found.\n\n"

        f"Expected:\n"
        f"{CLASS_NAMES_PATH}\n"

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
# LOAD CLASS NAMES
# ============================================================

print(
    "Loading class names..."
)


CLASS_NAMES = torch.load(

    CLASS_NAMES_PATH,

    map_location="cpu",

    weights_only=False

)


# ============================================================
# NORMALIZE CLASS NAMES
# ============================================================

if isinstance(
    CLASS_NAMES,
    tuple
):

    CLASS_NAMES = list(
        CLASS_NAMES
    )


elif not isinstance(
    CLASS_NAMES,
    list
):

    CLASS_NAMES = list(
        CLASS_NAMES
    )


# ============================================================
# NUMBER OF CLASSES
# ============================================================

NUM_CLASSES = len(
    CLASS_NAMES
)


print(
    "Number of classes:",
    NUM_CLASSES
)


if NUM_CLASSES != 38:

    raise RuntimeError(

        "\nIncorrect number of classes.\n"

        f"Expected: 38\n"
        f"Found: {NUM_CLASSES}\n"

    )


# ============================================================
# OPTIONAL CLASS MAPPING LOG
# ============================================================

print()
print("PlantAI class mapping:")

for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index:02d} : {class_name}"
    )

print()


# ============================================================
# GLOBAL MODEL
# ============================================================

# IMPORTANT:
#
# We DO NOT load the model at import time.
#
# It will be loaded only when the first prediction occurs.
#
# This reduces startup memory pressure.

model = None


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    """
    Create EfficientNet-B0 with 38 output classes.
    """

    model_instance = (
        models.efficientnet_b0(
            weights=None
        )
    )


    # --------------------------------------------------------
    # EfficientNet classifier
    # --------------------------------------------------------

    input_features = (
        model_instance
        .classifier[1]
        .in_features
    )


    model_instance.classifier[1] = (

        torch.nn.Linear(

            input_features,

            NUM_CLASSES

        )

    )


    return model_instance


# ============================================================
# CLEAN STATE DICTIONARY
# ============================================================

def clean_state_dict(
    state_dict
):

    """
    Remove common prefixes from saved checkpoints.

    Handles:

        module.xxx
        model.xxx
    """

    cleaned = {}


    for key, value in state_dict.items():

        # ----------------------------------------------------
        # DataParallel prefix
        # ----------------------------------------------------

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]


        # ----------------------------------------------------
        # Model prefix
        # ----------------------------------------------------

        if key.startswith(
            "model."
        ):

            key = key[
                len("model.") :
            ]


        cleaned[key] = value


    return cleaned


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    """
    Load EfficientNet model only once.

    Every request reuses the same model.
    """

    global model


    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if model is not None:

        return model


    print()
    print("=" * 60)
    print("Loading PlantAI model...")
    print("=" * 60)

    print(
        "Model path:",
        MODEL_PATH
    )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(

        MODEL_PATH,

        map_location="cpu",

        weights_only=False

    )


    # ========================================================
    # GET STATE DICTIONARY
    # ========================================================

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )


        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )


        else:

            state_dict = checkpoint


    else:

        state_dict = checkpoint


    # ========================================================
    # CREATE ARCHITECTURE
    # ========================================================

    model_instance = (
        create_model()
    )


    # ========================================================
    # CLEAN CHECKPOINT KEYS
    # ========================================================

    state_dict = (
        clean_state_dict(
            state_dict
        )
    )


    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    try:

        model_instance.load_state_dict(

            state_dict,

            strict=True

        )

    except RuntimeError as e:

        print()
        print("=" * 60)
        print("MODEL LOADING ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)

        raise


    # ========================================================
    # CPU
    # ========================================================

    model_instance = (
        model_instance.cpu()
    )


    # ========================================================
    # EVALUATION MODE
    # ========================================================

    model_instance.eval()


    # ========================================================
    # STORE GLOBAL MODEL
    # ========================================================

    model = model_instance


    # ========================================================
    # DELETE CHECKPOINT OBJECTS
    # ========================================================

    del checkpoint
    del state_dict
    del model_instance


    # ========================================================
    # GARBAGE COLLECTION
    # ========================================================

    gc.collect()


    print(
        "PlantAI model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print("=" * 60)
    print()


    return model


# ============================================================
# GET MODEL
# ============================================================

def get_model():

    return load_model()


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

def predict_image(
    image_path
):

    """
    Run PlantAI prediction.

    Returns:

        prediction
        confidence
        top_predictions
        confidence_status
    """

    print()
    print("=" * 60)
    print("Starting PlantAI prediction")
    print("=" * 60)

    print(
        "Image:",
        image_path
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    prediction_model = (
        load_model()
    )


    # ========================================================
    # IMAGE
    # ========================================================

    image = None

    image_tensor = None
    output = None
    probabilities = None
    top_probabilities = None
    top_indices = None


    try:

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            image_path
        )


        # ====================================================
        # CONVERT RGB
        # ====================================================

        image = image.convert(
            "RGB"
        )


        # ====================================================
        # RESIZE
        # ====================================================

        image = image.resize(
            (224, 224)
        )


        # ====================================================
        # TRANSFORM
        # ====================================================

        image_tensor = transform(
            image
        )


        # ====================================================
        # ADD BATCH DIMENSION
        # ====================================================

        image_tensor = (
            image_tensor
            .unsqueeze(0)
        )


        # ====================================================
        # CPU
        # ====================================================

        image_tensor = (
            image_tensor.cpu()
        )


        # ====================================================
        # INFERENCE
        # ====================================================

        with torch.inference_mode():

            output = prediction_model(
                image_tensor
            )


            probabilities = (
                torch.softmax(

                    output,

                    dim=1

                )
            )


            # =================================================
            # TOP 5
            # =================================================

            top_k = min(
                5,
                NUM_CLASSES
            )


            top_probabilities, top_indices = (

                torch.topk(

                    probabilities,

                    k=top_k,

                    dim=1

                )

            )


        # ====================================================
        # BUILD TOP PREDICTIONS
        # ====================================================

        top_predictions = []


        for probability, index in zip(

            top_probabilities[0],

            top_indices[0]

        ):

            index = int(
                index.item()
            )


            confidence_value = (

                float(
                    probability.item()
                )

                * 100.0

            )


            class_name = (
                CLASS_NAMES[index]
            )


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


        # ====================================================
        # BEST PREDICTION
        # ====================================================

        prediction = (
            top_predictions[0]
            ["class_name"]
        )


        confidence = (
            top_predictions[0]
            ["confidence"]
        )


        # ====================================================
        # CONFIDENCE STATUS
        # ====================================================

        if confidence < 40:

            confidence_status = (
                "Unknown"
            )


        elif confidence < 60:

            confidence_status = (
                "Low confidence"
            )


        elif confidence < 80:

            confidence_status = (
                "Possible diagnosis"
            )


        elif confidence < 95:

            confidence_status = (
                "Likely diagnosis"
            )


        else:

            confidence_status = (
                "High confidence"
            )


        # ====================================================
        # LOG
        # ====================================================

        print()
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

        print()
        print(
            "Top 5 predictions:"
        )


        for i, item in enumerate(

            top_predictions,

            start=1

        ):

            print(

                f"{i}. "
                f"{item['class_name']} "
                f"({item['confidence']:.2f}%)"

            )


        print(
            "=" * 60
        )


        # ====================================================
        # RETURN
        # ====================================================

        return (

            prediction,

            confidence,

            top_predictions,

            confidence_status

        )


    finally:

        # ====================================================
        # CLOSE PIL IMAGE
        # ====================================================

        if image is not None:

            try:

                image.close()

            except Exception:

                pass


        # ====================================================
        # DELETE TEMPORARY TENSORS
        # ====================================================

        image_tensor = None
        output = None
        probabilities = None
        top_probabilities = None
        top_indices = None


        # ====================================================
        # GARBAGE COLLECTION
        # ====================================================

        gc.collect()


        print(
            "Prediction temporary memory released."
        )


# ============================================================
# OPTIONAL MEMORY CLEANUP
# ============================================================

def cleanup():

    """
    Manually release temporary PyTorch memory.

    The model itself is intentionally kept loaded so
    subsequent predictions don't reload it.
    """

    gc.collect()


# ============================================================
# STARTUP MESSAGE
# ============================================================

print(
    "PlantAI prediction engine ready."
)

print(
    "Model will be loaded on first prediction."
)

print()