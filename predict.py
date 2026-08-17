# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Prediction Engine
============================================================

Render / 512 MB optimized version.

Features:
    - EfficientNet-B0
    - 38 plant disease classes
    - One prediction model
    - CPU optimized
    - Memory optimized
    - Top 5 predictions
    - Confidence status
    - No Grad-CAM
    - No attention model
    - No second AI model
    - Aggressive temporary memory cleanup

IMPORTANT:
    app.py expects these functions:

        predict_image()
        get_model()
        get_transform()
        cleanup()
============================================================
"""

import os
import gc

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CPU / DEVICE CONFIGURATION
# ============================================================

# Render free instances are CPU based.
# Limiting threads reduces memory usage and CPU contention.

try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("PlantAI Prediction Engine")
print("=" * 60)
print("Using device:", DEVICE)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_model_38class.pth"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [

    "Apple___Apple_scab",

    "Apple___Black_rot",

    "Apple___Cedar_apple_rust",

    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",

    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn_(maize)___Common_rust_",

    "Corn_(maize)___Northern_Leaf_Blight",

    "Corn_(maize)___healthy",

    "Grape___Black_rot",

    "Grape___Esca_(Black_Measles)",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",

    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",

    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",

    "Pepper,_bell___healthy",

    "Potato___Early_blight",

    "Potato___Late_blight",

    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",

    "Strawberry___healthy",

    "Tomato___Bacterial_spot",

    "Tomato___Early_blight",

    "Tomato___Late_blight",

    "Tomato___Leaf_Mold",

    "Tomato___Septoria_leaf_spot",

    "Tomato___Spider_mites Two-spotted_spider_mite",

    "Tomato___Target_Spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato___Tomato_mosaic_virus",

    "Tomato___healthy"
]


NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# GLOBAL MODEL
# ============================================================

MODEL = None


# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([

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
# CREATE MODEL
# ============================================================

def create_model():

    print(
        "Creating EfficientNet-B0..."
    )

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = torch.nn.Linear(

        model.classifier[1].in_features,

        NUM_CLASSES

    )

    return model


# ============================================================
# EXTRACT STATE DICT
# ============================================================

def extract_state_dict(checkpoint):

    # --------------------------------------------------------
    # Direct state_dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        # Common checkpoint format

        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]


        # Another common format

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]


        # Sometimes saved as:
        #
        # {
        #     "model": state_dict
        # }

        if "model" in checkpoint:

            model_value = checkpoint["model"]

            if isinstance(
                model_value,
                dict
            ):

                return model_value


        # Check whether checkpoint itself
        # looks like a state dictionary.

        tensor_count = 0

        for value in checkpoint.values():

            if torch.is_tensor(value):

                tensor_count += 1


        if tensor_count > 0:

            return checkpoint


    # --------------------------------------------------------
    # Complete PyTorch model
    # --------------------------------------------------------

    if hasattr(
        checkpoint,
        "state_dict"
    ):

        return checkpoint.state_dict()


    raise RuntimeError(
        "Could not find model state_dict "
        "inside checkpoint."
    )


# ============================================================
# CLEAN STATE DICT
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key


        # Remove DataParallel prefix

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]


        cleaned[new_key] = value


    return cleaned


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    """
    Loads checkpoint with Render-friendly settings.

    mmap=True can reduce the temporary memory pressure
    for compatible PyTorch checkpoint files.

    Falls back automatically if mmap is unavailable.
    """

    print(
        "Loading checkpoint..."
    )

    print(
        "Model path:",
        MODEL_PATH
    )


    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(

            "Model file not found:\n"

            + MODEL_PATH

        )


    # --------------------------------------------------------
    # Try memory-mapped loading first
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(

            MODEL_PATH,

            map_location="cpu",

            weights_only=False,

            mmap=True

        )

        print(
            "Checkpoint loaded using mmap."
        )

        return checkpoint


    except TypeError:

        # Older PyTorch versions don't support mmap
        pass


    except RuntimeError as e:

        print(
            "mmap loading unavailable:",
            e
        )


    # --------------------------------------------------------
    # Normal fallback
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(

            MODEL_PATH,

            map_location="cpu",

            weights_only=False

        )

    except TypeError:

        # Older PyTorch versions

        checkpoint = torch.load(

            MODEL_PATH,

            map_location="cpu"

        )


    print(
        "Checkpoint loaded."
    )


    return checkpoint


# ============================================================
# GET MODEL
# ============================================================

def get_model():

    global MODEL


    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if MODEL is not None:

        return MODEL


    print("=" * 60)

    print(
        "Loading PlantAI model..."
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(

            "PlantAI model file not found:\n"

            + MODEL_PATH

        )


    checkpoint = None

    state_dict = None

    model = None


    try:

        # ====================================================
        # CREATE ARCHITECTURE
        # ====================================================

        model = create_model()


        # ====================================================
        # LOAD CHECKPOINT
        # ====================================================

        checkpoint = load_checkpoint()


        # ====================================================
        # EXTRACT STATE DICT
        # ====================================================

        state_dict = extract_state_dict(
            checkpoint
        )


        # ====================================================
        # CLEAN KEYS
        # ====================================================

        state_dict = clean_state_dict(
            state_dict
        )


        # ====================================================
        # LOAD MODEL WEIGHTS
        # ====================================================

        try:

            model.load_state_dict(

                state_dict,

                strict=True

            )

            print(
                "Strict model loading successful."
            )


        except RuntimeError as strict_error:

            print(
                "Strict loading failed."
            )

            print(
                "Trying compatible loading..."
            )


            result = model.load_state_dict(

                state_dict,

                strict=False

            )


            print(
                "Missing keys:",
                result.missing_keys
            )

            print(
                "Unexpected keys:",
                result.unexpected_keys
            )


            # If important model weights are missing,
            # do not continue with an invalid model.

            if len(
                result.missing_keys
            ) > 0:

                raise RuntimeError(

                    "PlantAI model checkpoint does not "
                    "match EfficientNet-B0 38-class model.\n\n"

                    + str(strict_error)

                )


        # ====================================================
        # IMMEDIATELY RELEASE CHECKPOINT
        # ====================================================

        checkpoint = None

        state_dict = None

        gc.collect()


        # ====================================================
        # CPU MODEL
        # ====================================================

        model.to(
            DEVICE
        )


        # ====================================================
        # EVALUATION MODE
        # ====================================================

        model.eval()


        # ====================================================
        # SAVE GLOBAL MODEL
        # ====================================================

        MODEL = model

        model = None


        print("=" * 60)

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


        return MODEL


    except Exception:

        print(
            "MODEL LOAD FAILED"
        )


        traceback_message = None

        try:

            import traceback

            traceback_message = traceback.format_exc()

        except Exception:

            pass


        if traceback_message:

            print(
                traceback_message
            )


        # Release temporary objects

        checkpoint = None

        state_dict = None

        model = None

        gc.collect()


        raise


# ============================================================
# GET TRANSFORM
# ============================================================

def get_transform():

    return TRANSFORM


# ============================================================
# FORMAT PREDICTION NAME
# ============================================================

def format_prediction_name(
    class_name
):

    if not class_name:

        return "Unknown"


    # --------------------------------------------------------
    # Separate plant and disease
    # --------------------------------------------------------

    if "___" in class_name:

        plant, disease = class_name.split(
            "___",
            1
        )

    else:

        plant = "Unknown"

        disease = class_name


    # --------------------------------------------------------
    # Clean plant name
    # --------------------------------------------------------

    plant = plant.replace(
        "_",
        " "
    )


    plant = plant.replace(
        ",",
        ", "
    )


    # --------------------------------------------------------
    # Clean disease name
    # --------------------------------------------------------

    disease = disease.replace(
        "_",
        " "
    )


    while "  " in disease:

        disease = disease.replace(
            "  ",
            " "
        )


    return (

        plant.strip()

        + " - "

        + disease.strip()

    )


# ============================================================
# CONFIDENCE STATUS
# ============================================================

def get_confidence_status(
    confidence
):

    confidence = float(
        confidence
    )


    if confidence >= 85:

        return "Very High Confidence"


    if confidence >= 70:

        return "High Confidence"


    if confidence >= 55:

        return "Moderate Confidence"


    if confidence >= 40:

        return "Low Confidence"


    return "Very Low Confidence"


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image_path
):

    """
    Predict plant disease.

    Returns:

        prediction
        confidence
        top_predictions
        confidence_status
    """

    # ========================================================
    # GET SINGLE GLOBAL MODEL
    # ========================================================

    model = get_model()


    # ========================================================
    # VALIDATE FILE
    # ========================================================

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(

            "Image not found: "

            + str(image_path)

        )


    image = None

    tensor = None

    outputs = None

    probabilities = None

    values = None

    indices = None


    try:

        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # ====================================================
        # RESIZE / TRANSFORM
        # ====================================================

        tensor = TRANSFORM(
            image
        )


        tensor = tensor.unsqueeze(
            0
        )


        tensor = tensor.to(
            DEVICE
        )


        # ====================================================
        # INFERENCE
        # ====================================================

        with torch.inference_mode():

            outputs = model(
                tensor
            )


            probabilities = torch.softmax(

                outputs,

                dim=1

            )


            # =================================================
            # TOP 5
            # =================================================

            top_count = min(
                5,
                NUM_CLASSES
            )


            values, indices = torch.topk(

                probabilities,

                top_count,

                dim=1

            )


        # ====================================================
        # MOVE ONLY TOP RESULTS TO CPU
        # ====================================================

        values_cpu = (

            values[0]
            .detach()
            .cpu()
            .tolist()

        )


        indices_cpu = (

            indices[0]
            .detach()
            .cpu()
            .tolist()

        )


        # ====================================================
        # BEST PREDICTION
        # ====================================================

        best_index = int(
            indices_cpu[0]
        )


        confidence = (

            float(
                values_cpu[0]
            )

            * 100.0

        )


        raw_prediction = CLASS_NAMES[
            best_index
        ]


        prediction = format_prediction_name(
            raw_prediction
        )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        top_predictions = []


        for value, index in zip(

            values_cpu,

            indices_cpu

        ):

            index = int(
                index
            )


            raw_name = CLASS_NAMES[
                index
            ]


            formatted_name = (
                format_prediction_name(
                    raw_name
                )
            )


            top_predictions.append({

                "class":
                    formatted_name,

                "raw_class":
                    raw_name,

                "confidence":
                    round(

                        float(value)
                        * 100.0,

                        2

                    )

            })


        # ====================================================
        # CONFIDENCE STATUS
        # ====================================================

        confidence_status = (
            get_confidence_status(
                confidence
            )
        )


        # ====================================================
        # LOG RESULT
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

        print(
            "Top predictions:",
            top_predictions
        )


        # ====================================================
        # RETURN
        # ====================================================

        return (

            prediction,

            float(
                confidence
            ),

            top_predictions,

            confidence_status

        )


    finally:

        # ====================================================
        # RELEASE TEMPORARY PYTORCH OBJECTS
        # ====================================================

        image = None

        tensor = None

        outputs = None

        probabilities = None

        values = None

        indices = None


        # ====================================================
        # GARBAGE COLLECTION
        # ====================================================

        gc.collect()


        # ====================================================
        # CUDA CLEANUP
        # ====================================================

        if DEVICE.type == "cuda":

            try:

                torch.cuda.empty_cache()

            except Exception:

                pass


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    """
    Cleanup temporary memory.

    IMPORTANT:
    We intentionally DO NOT delete MODEL.

    The model is kept globally so Render does not have
    to reload EfficientNet for every request.
    """

    gc.collect()


    if DEVICE.type == "cuda":

        try:

            torch.cuda.empty_cache()

        except Exception:

            pass


# ============================================================
# MODEL STATUS
# ============================================================

def model_loaded():

    return MODEL is not None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("PlantAI Prediction Engine Test")
    print("=" * 60)

    print(
        "Model path:",
        MODEL_PATH
    )

    print(
        "Model exists:",
        os.path.exists(
            MODEL_PATH
        )
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "CPU threads:",
        torch.get_num_threads()
    )


    try:

        model = get_model()

        print()
        print(
            "MODEL TEST SUCCESSFUL"
        )

        print(
            "Model loaded:",
            model is not None
        )

    except Exception as e:

        print()
        print(
            "MODEL TEST FAILED"
        )

        print(
            str(e)
        )


    print("=" * 60)