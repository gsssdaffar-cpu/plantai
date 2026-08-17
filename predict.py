# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Prediction Engine
============================================================

Render optimized prediction engine.

Features:
    - EfficientNet-B0
    - 38 plant disease classes
    - One model only
    - CPU compatible
    - Memory optimized
    - Top predictions
    - Confidence status
    - No Grad-CAM
    - No second AI model
============================================================
"""

import os
import gc

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# MODEL
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
# MODEL CREATION
# ============================================================

def create_model():

    print(
        "Creating EfficientNet-B0..."
    )

    model = models.efficientnet_b0(
        weights=None
    )

    # Replace classifier
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# CHECKPOINT EXTRACTION
# ============================================================

def extract_state_dict(checkpoint):

    # --------------------------------------------------------
    # Direct state_dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if "model" in checkpoint:

            model_value = checkpoint["model"]

            if isinstance(
                model_value,
                dict
            ):

                return model_value


        # Check if this itself looks like state_dict
        keys = list(
            checkpoint.keys()
        )

        if keys:

            tensor_count = sum(

                1

                for value in checkpoint.values()

                if torch.is_tensor(value)

            )

            if tensor_count > 0:

                return checkpoint


    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    if hasattr(
        checkpoint,
        "state_dict"
    ):

        return checkpoint.state_dict()


    raise RuntimeError(
        "Could not find model state_dict in checkpoint."
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
# LOAD MODEL
# ============================================================

def get_model():

    global MODEL

    if MODEL is not None:

        return MODEL


    print(
        "Loading PlantAI model..."
    )

    print(
        "Model path:",
        MODEL_PATH
    )


    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(

            "Model file not found: "

            + MODEL_PATH

        )


    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = create_model()


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    try:

        checkpoint = torch.load(

            MODEL_PATH,

            map_location=DEVICE,

            weights_only=False

        )

    except TypeError:

        # Older PyTorch versions
        checkpoint = torch.load(

            MODEL_PATH,

            map_location=DEVICE

        )


    # --------------------------------------------------------
    # Extract weights
    # --------------------------------------------------------

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError as e:

        print(
            "Strict model loading failed."
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

        if len(result.missing_keys) > 0:

            raise RuntimeError(

                "Model checkpoint does not match "
                "EfficientNet-B0 38-class architecture.\n"
                + str(e)

            )


    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Store globally
    # --------------------------------------------------------

    MODEL = model


    print(
        "PlantAI model loaded successfully."
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Device:",
        DEVICE
    )


    return MODEL


# ============================================================
# GET TRANSFORM
# ============================================================

def get_transform():

    return TRANSFORM


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_prediction_name(
    class_name
):

    if not class_name:

        return "Unknown"


    # Split plant / disease
    if "___" in class_name:

        plant, disease = class_name.split(
            "___",
            1
        )

    else:

        plant = "Unknown"

        disease = class_name


    # Clean plant name
    plant = plant.replace(
        "_",
        " "
    )

    plant = plant.replace(
        ",",
        ", "
    )

    # Clean disease name
    disease = disease.replace(
        "_",
        " "
    )

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
# PREDICTION
# ============================================================

def predict_image(
    image_path
):

    model = get_model()


    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    image = None

    tensor = None

    probabilities = None


    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        tensor = TRANSFORM(
            image
        ).unsqueeze(
            0
        )


        tensor = tensor.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


        # ----------------------------------------------------
        # Top predictions
        # ----------------------------------------------------

        top_count = min(
            5,
            NUM_CLASSES
        )


        values, indices = torch.topk(

            probabilities,

            top_count,

            dim=1

        )


        values = values[
            0
        ].detach().cpu().tolist()


        indices = indices[
            0
        ].detach().cpu().tolist()


        # ----------------------------------------------------
        # Best prediction
        # ----------------------------------------------------

        best_index = indices[0]

        confidence = (
            values[0] * 100.0
        )


        raw_prediction = CLASS_NAMES[
            best_index
        ]


        prediction = format_prediction_name(
            raw_prediction
        )


        # ----------------------------------------------------
        # Top prediction list
        # ----------------------------------------------------

        top_predictions = []


        for value, index in zip(
            values,
            indices
        ):

            raw_name = CLASS_NAMES[
                index
            ]

            formatted_name = format_prediction_name(
                raw_name
            )


            top_predictions.append({

                "class":
                    formatted_name,

                "raw_class":
                    raw_name,

                "confidence":
                    round(
                        value * 100.0,
                        2
                    )

            })


        # ----------------------------------------------------
        # Confidence status
        # ----------------------------------------------------

        confidence_status = get_confidence_status(
            confidence
        )


        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        print(
            "Status:",
            confidence_status
        )


        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return (

            prediction,

            float(
                confidence
            ),

            top_predictions,

            confidence_status

        )


    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        image = None

        tensor = None

        probabilities = None

        try:

            del outputs

        except Exception:

            pass

        try:

            del values

        except Exception:

            pass

        try:

            del indices

        except Exception:

            pass

        gc.collect()


        if DEVICE.type == "cuda":

            try:

                torch.cuda.empty_cache()

            except Exception:

                pass


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    gc.collect()


    if DEVICE.type == "cuda":

        try:

            torch.cuda.empty_cache()

        except Exception:

            pass


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("PlantAI Prediction Engine Test")
    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Classes:",
        NUM_CLASSES
    )

    print(
        "Device:",
        DEVICE
    )

    model = get_model()

    print(
        "Model test successful."
    )

    print("=" * 60)