# -*- coding: utf-8 -*-
from pathlib import Path

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
        f"{MODEL_PATH}\n\n"
        f"Expected location:\n"
        f"{BASE_DIR.parent / 'models'}"
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
    map_location=DEVICE,
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
        "Expected class_names.pth at:\n"
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
# CREATE EFFICIENTNET-B0
# ============================================================

def create_model():

    model = models.efficientnet_b0(
        weights=None
    )


    # --------------------------------------------------------
    # Replace classifier
    # --------------------------------------------------------

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

    # Remove DataParallel prefix

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]


    # Remove model prefix

    if key.startswith("model."):

        key = key[
            len("model.") :
        ]


    clean_state_dict[key] = value


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
    print("❌ MODEL LOADING ERROR")
    print("=" * 60)
    print(e)
    print("=" * 60)

    raise


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
# MODEL INFORMATION
# ============================================================

if isinstance(checkpoint, dict):

    if "best_accuracy" in checkpoint:

        print(
            f"\nBest validation accuracy: "
            f"{checkpoint['best_accuracy']:.2f}%"
        )

    if "num_classes" in checkpoint:

        print(
            f"Model classes: "
            f"{checkpoint['num_classes']}"
        )


print()
print(
    "✅ Prediction model loaded:"
)

print(
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

# ============================================================
# PREDICT IMAGE - TOP 5
# ============================================================

def predict_image(image_path):

    """
    Predict plant disease.

    Returns:

        prediction
        confidence
        top_predictions

    top_predictions format:

        [
            {
                "class_name": "...",
                "confidence": 62.94,
                "index": 7
            }
        ]
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .unsqueeze(0)
    )


    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )


        # ----------------------------------------------------
        # TOP 5
        # ----------------------------------------------------

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(5, len(CLASS_NAMES)),
            dim=1
        )


    # --------------------------------------------------------
    # Convert TOP 5
    # --------------------------------------------------------

    top_predictions = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        index = index.item()

        confidence_value = (
            probability.item() * 100
        )

        class_name = CLASS_NAMES[index]

        top_predictions.append({

            "class_name": class_name,

            "confidence": round(
                confidence_value,
                2
            ),

            "index": index

        })


    # --------------------------------------------------------
    # BEST PREDICTION
    # --------------------------------------------------------

    prediction = (
        top_predictions[0]["class_name"]
    )

    confidence = (
        top_predictions[0]["confidence"]
    )


    # --------------------------------------------------------
    # CONFIDENCE STATUS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("🌱 PLANTAI PREDICTION")
    print("=" * 60)

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        "Status:",
        confidence_status
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


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return (
        prediction,
        confidence,
        top_predictions,
        confidence_status
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Transform image
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .unsqueeze(0)
    )


    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    image_tensor = (
        image_tensor
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    # --------------------------------------------------------
    # Convert values
    # --------------------------------------------------------

    predicted_index = (
        predicted_index.item()
    )


    confidence = (
        confidence.item()
        * 100
    )


    # --------------------------------------------------------
    # Get class name
    # --------------------------------------------------------

    if (
        0 <= predicted_index
        < len(CLASS_NAMES)
    ):

        prediction = CLASS_NAMES[
            predicted_index
        ]

    else:

        prediction = "Unknown"


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )


    return (
        prediction,
        confidence
    )