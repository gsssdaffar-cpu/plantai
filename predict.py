# -*- coding: utf-8 -*-

from pathlib import Path
import gc

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# DEVICE
# ============================================================

# Render normally runs without CUDA.
# Force CPU when CUDA is unavailable.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# CPU MEMORY OPTIMIZATION
# ============================================================

# These settings help reduce unnecessary CPU threading/memory.
# Only configure threads when running on CPU.

if DEVICE.type == "cpu":

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
        "\nModel file not found:\n"
        f"{MODEL_PATH}\n\n"
        "Expected location:\n"
        f"{BASE_DIR / 'models'}"
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


checkpoint = torch.load(

    MODEL_PATH,

    map_location="cpu",

    weights_only=False

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

        "\nClass names not found.\n"

        "Expected:\n"

        f"{CLASS_NAMES_PATH}"

    )


# ============================================================
# ENSURE CLASS NAMES ARE A LIST
# ============================================================

if isinstance(
    CLASS_NAMES,
    tuple
):

    CLASS_NAMES = list(
        CLASS_NAMES
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

        f"\nExpected 38 classes, "
        f"but found {NUM_CLASSES}."

    )


# ============================================================
# PRINT CLASS MAPPING
# ============================================================

print()
print("Class mapping:")

for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index:02d} : {class_name}"
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = models.efficientnet_b0(

        weights=None

    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = (
        torch.nn.Linear(

            input_features,

            NUM_CLASSES

        )
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

    # Remove DataParallel prefix

    if key.startswith(
        "module."
    ):

        key = key[
            len("module.") :
        ]


    # Remove model prefix

    if key.startswith(
        "model."
    ):

        key = key[
            len("model.") :
        ]


    clean_state_dict[
        key
    ] = value


# ============================================================
# RELEASE CHECKPOINT REFERENCES
# ============================================================

# We no longer need the original checkpoint
# after extracting the weights and classes.

del checkpoint
del state_dict

gc.collect()


# ============================================================
# LOAD WEIGHTS
# ============================================================

try:

    model.load_state_dict(

        clean_state_dict,

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


# Release temporary state dictionary
del clean_state_dict

gc.collect()


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# DISABLE GRADIENTS FOR MODEL
# ============================================================

# The prediction endpoint never needs
# gradients. This reduces memory usage.

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print(
    "PlantAI model loaded successfully."
)

print(
    "Model:",
    "EfficientNet-B0"
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
    "Model path:",
    MODEL_PATH
)

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

    Example:

        (
            "Tomato___Early_blight",
            91.42,
            [
                {
                    "class_name": "Tomato___Early_blight",
                    "confidence": 91.42,
                    "index": 4
                }
            ],
            "Likely diagnosis"
        )
    """

    image = None
    image_tensor = None
    output = None
    probabilities = None
    top_probabilities = None
    top_indices = None


    try:

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # ====================================================
        # TRANSFORM IMAGE
        # ====================================================

        image_tensor = transform(
            image
        )


        # ====================================================
        # RELEASE PIL IMAGE
        # ====================================================

        # The tensor now contains the image data.

        image.close()

        image = None


        # ====================================================
        # ADD BATCH DIMENSION
        # ====================================================

        image_tensor = (
            image_tensor
            .unsqueeze(0)
        )


        # ====================================================
        # MOVE TO DEVICE
        # ====================================================

        image_tensor = (
            image_tensor.to(
                DEVICE
            )
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        with torch.inference_mode():

            output = model(
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


            # =================================================
            # COPY ONLY SMALL RESULTS TO CPU
            # =================================================

            top_probabilities_cpu = (
                top_probabilities[0]
                .detach()
                .cpu()
                .tolist()
            )


            top_indices_cpu = (
                top_indices[0]
                .detach()
                .cpu()
                .tolist()
            )


        # ====================================================
        # CONVERT TOP 5
        # ====================================================

        top_predictions = []


        for probability, index in zip(

            top_probabilities_cpu,

            top_indices_cpu

        ):

            confidence_value = (
                float(probability)
                * 100.0
            )


            class_name = CLASS_NAMES[
                int(index)
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
                    int(index)

            })


        # ====================================================
        # BEST PREDICTION
        # ====================================================

        prediction = (
            top_predictions[0][
                "class_name"
            ]
        )


        confidence = (
            top_predictions[0][
                "confidence"
            ]
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
        # PRINT RESULT
        # ====================================================

        print()
        print("=" * 60)
        print("PLANTAI PREDICTION")
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


        print("=" * 60)
        print()


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
        # MEMORY CLEANUP
        # ====================================================

        # Important for Render's 512 MB limit.

        try:

            if image is not None:

                image.close()

        except Exception:

            pass


        # Delete temporary tensors.

        image_tensor = None
        output = None
        probabilities = None
        top_probabilities = None
        top_indices = None


        # Python garbage collection.

        gc.collect()


        # CUDA cleanup only when CUDA exists.

        if DEVICE.type == "cuda":

            try:

                torch.cuda.empty_cache()

            except Exception:

                pass