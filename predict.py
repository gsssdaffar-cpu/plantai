# ============================================================
# PLANTAI
# V2 PRODUCTION PREDICTION ENGINE
# 38-CLASS PLANTVILLAGE DISEASE DETECTION
# ============================================================

import os
import sys
import gc

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from model import create_model
from config import DEVICE


# ============================================================
# PRODUCTION CPU OPTIMIZATION
# ============================================================

# Render production instances have limited CPU/RAM resources.
# Restrict PyTorch CPU thread pools to avoid excessive
# thread creation and memory pressure.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_model_38class_V2.pth"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names_V2.pth"
)

IMAGE_SIZE = 224
TOP_K = 5

# TTA is only performed when confidence is below this value.
TTA_THRESHOLD = 60.0


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()
print("============================================================")
print("Loading PlantAI V2 model...")
print("============================================================")
print("Device:", DEVICE)
print("Model:", MODEL_PATH)
print("Class names:", CLASS_NAMES_PATH)
print("PyTorch CPU threads:", torch.get_num_threads())
print("PyTorch interop threads:", torch.get_num_interop_threads())
print("============================================================")


# ============================================================
# LOAD V2 CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_NAMES_PATH):

    raise FileNotFoundError(
        "\nPlantAI V2 class file not found:\n"
        f"{CLASS_NAMES_PATH}\n"
    )


try:

    CLASS_NAMES = torch.load(
        CLASS_NAMES_PATH,
        map_location="cpu",
        weights_only=False
    )

except TypeError:

    # Compatibility fallback
    CLASS_NAMES = torch.load(
        CLASS_NAMES_PATH,
        map_location="cpu"
    )


# ============================================================
# VERIFY CLASS NAMES
# ============================================================

if not isinstance(CLASS_NAMES, list):

    raise TypeError(
        "class_names_V2.pth must contain a Python list."
    )


if len(CLASS_NAMES) != 38:

    raise ValueError(
        "PlantAI V2 requires exactly 38 classes.\n"
        f"Found: {len(CLASS_NAMES)}"
    )


NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", NUM_CLASSES)


# ============================================================
# DISPLAY CLASS NAMES
# ============================================================

print()
print("V2 CLASS NAMES")
print("------------------------------------------------------------")

for index, class_name in enumerate(CLASS_NAMES):

    print(
        f"{index:02d} : {class_name}"
    )

print("------------------------------------------------------------")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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

print()
print("============================================================")
print("Creating EfficientNet-B0...")
print("============================================================")

model = create_model(
    NUM_CLASSES
)


# ============================================================
# VERIFY MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(

        "\nPlantAI V2 model not found:\n"

        f"{MODEL_PATH}\n\n"

        "Expected file:\n"

        "models\\plant_model_38class_V2.pth"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")
print("Model path:", MODEL_PATH)

checkpoint = None


try:

    checkpoint = torch.load(

        MODEL_PATH,

        map_location="cpu",

        mmap=True,

        weights_only=False
    )

    print(
        "Checkpoint loaded using mmap."
    )


except TypeError:

    checkpoint = torch.load(

        MODEL_PATH,

        map_location="cpu"
    )

    print(
        "Checkpoint loaded using standard torch.load."
    )


except Exception as e:

    print()
    print("mmap loading failed.")
    print("Reason:", str(e))
    print(
        "Trying standard checkpoint loading..."
    )

    checkpoint = torch.load(

        MODEL_PATH,

        map_location="cpu",

        weights_only=False
    )

    print(
        "Checkpoint loaded using standard loading."
    )


# ============================================================
# HANDLE CHECKPOINT FORMAT
# ============================================================

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

        print(
            "Checkpoint format: state_dict wrapper."
        )

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "Checkpoint format: "
            "model_state_dict wrapper."
        )

    else:

        state_dict = checkpoint

        print(
            "Checkpoint format: "
            "direct state_dict."
        )

else:

    raise TypeError(
        "Unsupported model checkpoint format."
    )


# ============================================================
# REMOVE POSSIBLE 'module.' PREFIX
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    clean_key = (

        key[7:]

        if key.startswith("module.")

        else key
    )

    clean_state_dict[
        clean_key
    ] = value


# ============================================================
# RELEASE CHECKPOINT REFERENCE
# ============================================================

# The state dictionary has already been copied into the
# cleaned dictionary. We can release the original checkpoint
# reference before inference.
del state_dict
del checkpoint

gc.collect()


# ============================================================
# STRICT MODEL LOADING
# ============================================================

try:

    model.load_state_dict(
        clean_state_dict,
        strict=True
    )

    print(
        "Strict model loading successful."
    )

except Exception as e:

    print()
    print("============================================================")
    print("STRICT MODEL LOADING FAILED")
    print("============================================================")
    print(str(e))
    print("============================================================")

    raise


# Release temporary state dictionary after loading.
del clean_state_dict

gc.collect()


# ============================================================
# MOVE MODEL TO DEVICE & SET EVAL MODE
# ============================================================

model.to(
    DEVICE
)

model.eval()


# ============================================================
# DISABLE GRADIENTS PERMANENTLY
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad_(
        False
    )


# ============================================================
# FINAL MODEL INFORMATION
# ============================================================

print()
print("============================================================")
print("PlantAI V2 model loaded successfully.")
print("============================================================")
print("Device:", DEVICE)
print("Classes:", NUM_CLASSES)
print("Model:", MODEL_PATH)
print(
    "CPU threads:",
    torch.get_num_threads()
)
print(
    "Interop threads:",
    torch.get_num_interop_threads()
)
print("============================================================")
print()


# ============================================================
# COMPATIBILITY & CLEANUP FUNCTIONS
# ============================================================

def get_model():

    return model


def get_classes():

    return CLASS_NAMES


def get_transform():

    return transform


def cleanup():

    """
    Frees CUDA memory cache and triggers garbage collection.
    """

    try:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    except Exception:
        pass

    gc.collect()


# ============================================================
# BASIC IMAGE PREPARATION
# ============================================================

def prepare_image(
    image_path
):

    if not image_path:

        raise ValueError(
            "Image path is empty."
        )


    if not os.path.isabs(
        image_path
    ):

        image_path = os.path.join(
            BASE_DIR,
            image_path
        )


    image_path = os.path.abspath(
        image_path
    )


    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(

            f"\nImage not found:\n"
            f"{image_path}\n\n"

            f"Current PlantAI directory:\n"
            f"{BASE_DIR}\n"
        )


    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as e:

        raise RuntimeError(

            f"Could not open image:\n"
            f"{image_path}\n\n"
            f"Error: {str(e)}"
        )


    return image


# ============================================================
# RAW MODEL PREDICTION WITH OPTIONAL TTA
# ============================================================

def get_probabilities(
    image_path
):

    image = prepare_image(
        image_path
    )


    # ========================================================
    # SINGLE PASS
    # ========================================================

    single_tensor = (

        transform(image)

        .unsqueeze(0)

        .to(DEVICE)
    )


    with torch.inference_mode():

        outputs = model(
            single_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )


    # ========================================================
    # RELEASE SINGLE-PASS TENSOR
    # ========================================================

    del outputs
    del single_tensor


    # ========================================================
    # CHECK CONFIDENCE
    # ========================================================

    top_prob, _ = torch.max(
        probabilities,
        dim=1
    )

    confidence_val = (
        top_prob.item()
        * 100.0
    )


    # ========================================================
    # OPTIONAL TEST-TIME AUGMENTATION
    # ========================================================

    if confidence_val < TTA_THRESHOLD:

        print(
            "[TTA] Low confidence detected "
            f"({confidence_val:.2f}%). "
            "Running Test-Time Augmentation..."
        )


        variants = [

            image,

            image.transpose(
                Image.FLIP_LEFT_RIGHT
            ),

            image.transpose(
                Image.FLIP_TOP_BOTTOM
            ),

            image.rotate(90),

            image.rotate(270)
        ]


        stacked_tensors = torch.stack(

            [
                transform(v)
                for v in variants
            ]

        ).to(
            DEVICE
        )


        with torch.inference_mode():

            tta_outputs = model(
                stacked_tensors
            )

            tta_probabilities = F.softmax(
                tta_outputs,
                dim=1
            ).mean(
                dim=0,
                keepdim=True
            )


        del tta_outputs
        del stacked_tensors

        probabilities = (
            tta_probabilities
        )


        new_prob, _ = torch.max(
            probabilities,
            dim=1
        )


        print(
            "[TTA] Adjusted confidence: "
            f"{new_prob.item() * 100.0:.2f}%"
        )


        del new_prob


    # ========================================================
    # CLEANUP
    # ========================================================

    del image

    gc.collect()


    return probabilities


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image_path
):

    """
    Returns:

        (
            prediction_full_name,
            confidence_value,
            top_predictions_list,
            confidence_status
        )
    """

    analysis = analyze_image(
        image_path
    )


    prediction = analysis[
        "full_name"
    ]

    confidence = analysis[
        "confidence"
    ]

    confidence_status = analysis[
        "confidence_level"
    ]


    top_preds_formatted = []


    for item in analysis[
        "top_predictions"
    ]:

        formatted = format_class_name(
            item["class"]
        )


        top_preds_formatted.append({

            "class":
                formatted[
                    "full_name"
                ],

            "raw_class":
                item["class"],

            "confidence":
                round(
                    item["confidence"],
                    2
                )
        })


    return (

        prediction,

        confidence,

        top_preds_formatted,

        confidence_status
    )


# ============================================================
# PREDICTION WITH DETAILS
# ============================================================

def predict_with_details(
    image_path
):

    probabilities = get_probabilities(
        image_path
    )


    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


    class_index = predicted.item()

    class_name = CLASS_NAMES[
        class_index
    ]


    confidence_value = (
        confidence.item()
        * 100.0
    )


    # ========================================================
    # TOP 5 PREDICTIONS
    # ========================================================

    top_count = min(
        TOP_K,
        NUM_CLASSES
    )


    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            top_count,
            dim=1
        )
    )


    top_predictions = []


    for probability, index in zip(

        top_probabilities[0],

        top_indices[0]
    ):

        top_predictions.append({

            "class":
                CLASS_NAMES[
                    index.item()
                ],

            "confidence":
                probability.item()
                * 100
        })


    # ========================================================
    # PREDICTION MARGIN
    # ========================================================

    if len(top_predictions) >= 2:

        prediction_margin = (

            top_predictions[0][
                "confidence"
            ]

            -

            top_predictions[1][
                "confidence"
            ]
        )

    else:

        prediction_margin = (
            top_predictions[0][
                "confidence"
            ]
        )


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    if confidence_value >= 85:

        confidence_level = (
            "High Confidence"
        )

    elif confidence_value >= 60:

        confidence_level = (
            "Moderate Confidence"
        )

    else:

        confidence_level = (
            "Low Confidence"
        )


    # ========================================================
    # CLEAN TEMPORARY TENSORS
    # ========================================================

    del probabilities
    del confidence
    del predicted
    del top_probabilities
    del top_indices

    gc.collect()


    return {

        "class_name":
            class_name,

        "confidence":
            confidence_value,

        "class_index":
            class_index,

        "prediction_margin":
            prediction_margin,

        "confidence_level":
            confidence_level,

        "top_predictions":
            top_predictions
    }


# ============================================================
# TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    image_path,
    top_k=5
):

    probabilities = get_probabilities(
        image_path
    )


    top_k = min(
        top_k,
        NUM_CLASSES
    )


    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            top_k,
            dim=1
        )
    )


    results = []


    for probability, index in zip(

        top_probabilities[0],

        top_indices[0]
    ):

        results.append({

            "class":
                CLASS_NAMES[
                    index.item()
                ],

            "confidence":
                probability.item()
                * 100
        })


    del probabilities
    del top_probabilities
    del top_indices

    gc.collect()


    return results


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(
    class_index
):

    if class_index < 0:

        raise ValueError(
            "Class index cannot be negative."
        )


    if class_index >= NUM_CLASSES:

        raise ValueError(
            f"Invalid class index: "
            f"{class_index}"
        )


    return CLASS_NAMES[
        class_index
    ]


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_class_name(
    class_name
):

    formatted = class_name


    if "___" in formatted:

        plant, disease = (
            formatted.split(
                "___",
                1
            )
        )

    else:

        plant = formatted
        disease = ""


    # ========================================================
    # CLEAN PLANT NAME
    # ========================================================

    plant = (

        plant
        .replace("_", " ")
        .strip()
    )


    # ========================================================
    # CLEAN DISEASE NAME
    # ========================================================

    disease = (

        disease
        .replace("_", " ")
        .replace("(", "")
        .replace(")", "")
    )


    disease = (
        " ".join(
            disease.split()
        )
        .strip()
    )


    return {

        "plant":
            plant,

        "disease":
            disease,

        "full_name":

            (
                f"{plant} - {disease}"

                if disease

                else plant
            )
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_image(
    image_path
):

    result = predict_with_details(
        image_path
    )


    formatted = format_class_name(
        result["class_name"]
    )


    return {

        "class_name":
            result["class_name"],

        "plant":
            formatted["plant"],

        "disease":
            formatted["disease"],

        "full_name":
            formatted["full_name"],

        "confidence":
            result["confidence"],

        "class_index":
            result["class_index"],

        "prediction_margin":
            result["prediction_margin"],

        "confidence_level":
            result["confidence_level"],

        "top_predictions":
            result["top_predictions"]
    }


# ============================================================
# DIRECT COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("============================================================")
    print("                  MODEL TEST")
    print("============================================================")

    print(
        "Model loaded:",
        model is not None
    )

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

    print(
        "CPU threads:",
        torch.get_num_threads()
    )

    print(
        "Interop threads:",
        torch.get_num_interop_threads()
    )

    print("============================================================")


    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "  python predict.py "
            "\"uploads\\2.jpg\"\n"
        )

        sys.exit(0)


    image_path = sys.argv[1]


    display_path = (

        os.path.abspath(
            image_path
        )

        if os.path.isabs(
            image_path
        )

        else os.path.abspath(

            os.path.join(
                BASE_DIR,
                image_path
            )
        )
    )


    print()
    print("============================================================")
    print("                  IMAGE PREDICTION TEST")
    print("============================================================")

    print(
        "Image:",
        image_path
    )

    print(
        "Resolved path:",
        display_path
    )

    print(
        "Image exists:",
        os.path.exists(
            display_path
        )
    )

    print("============================================================")


    try:

        import time

        start_time = time.time()


        result = analyze_image(
            image_path
        )


        elapsed = (
            time.time()
            - start_time
        )


        print()
        print("============================================================")
        print("                  PLANTAI V2 PREDICTION")
        print("============================================================")

        print(
            "Plant            :",
            result["plant"]
        )

        print(
            "Disease          :",
            result["disease"]
        )

        print(
            "Full Class       :",
            result["class_name"]
        )

        print(
            "Confidence       :",
            f"{result['confidence']:.2f}%"
        )

        print(
            "Confidence Level :",
            result["confidence_level"]
        )

        print(
            "Margin           :",
            f"{result['prediction_margin']:.2f}%"
        )

        print(
            "Inference Time   :",
            f"{elapsed:.3f} seconds"
        )

        print()
        print("Top Predictions:")
        print("------------------------------------------------------------")


        for index, item in enumerate(

            result[
                "top_predictions"
            ],

            start=1
        ):

            print(

                f"{index}. "
                f"{item['class']} : "
                f"{item['confidence']:.2f}%"
            )


        print("============================================================")
        print("                  TEST COMPLETED")
        print("============================================================")
        print()


    except Exception as e:

        print()
        print("============================================================")
        print("                  PREDICTION TEST FAILED")
        print("============================================================")

        print(
            str(e)
        )

        print("============================================================")
        print()

        raise
