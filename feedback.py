# -*- coding: utf-8 -*-

"""
============================================================
PlantAI CONTINUOUS LEARNING - FEEDBACK ENGINE
============================================================

Purpose:
    Store VERIFIED user corrections for future model training.

IMPORTANT:
    - AI predictions are NOT automatically added to training.
    - A sample only enters the verified dataset when a user
      explicitly confirms/corrects the diagnosis.
    - Production V2 model is never modified by this file.

Structure:

learning/
    incoming/
    verified/
        <class_name>/
            image files
    rejected/
    logs/
        feedback.csv

============================================================
"""

import os
import csv
import uuid
import shutil
from datetime import datetime

import torch
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LEARNING_DIR = os.path.join(
    BASE_DIR,
    "learning"
)

INCOMING_DIR = os.path.join(
    LEARNING_DIR,
    "incoming"
)

VERIFIED_DIR = os.path.join(
    LEARNING_DIR,
    "verified"
)

REJECTED_DIR = os.path.join(
    LEARNING_DIR,
    "rejected"
)

LOG_DIR = os.path.join(
    LEARNING_DIR,
    "logs"
)

FEEDBACK_LOG = os.path.join(
    LOG_DIR,
    "feedback.csv"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH_V2 = os.path.join(
    BASE_DIR,
    "models",
    "plant_model_38class_V2.pth"
)

CLASS_NAMES_PATH_V2 = os.path.join(
    BASE_DIR,
    "models",
    "class_names_V2.pth"
)

MODEL_PATH_V3_CANDIDATE = os.path.join(
    BASE_DIR,
    "models",
    "plant_model_38class_V3_candidate.pth"
)

MODEL_PATH_V3 = os.path.join(
    BASE_DIR,
    "models",
    "plant_model_38class_V3.pth"
)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_learning_directories():
    """
    Create all continuous-learning directories if they
    do not already exist.
    """

    directories = [
        LEARNING_DIR,
        INCOMING_DIR,
        VERIFIED_DIR,
        REJECTED_DIR,
        LOG_DIR
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# MODEL CLASS INFORMATION
# ============================================================

def get_model_classes():
    """
    Load the PlantAI V2 class names.

    Priority:
        1. Classes stored inside V2 checkpoint
        2. class_names_V2.pth fallback

    Returns:
        list[str]

    Raises:
        FileNotFoundError
        RuntimeError
    """

    classes = None

    # --------------------------------------------------------
    # Check V2 model
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH_V2):

        checkpoint = torch.load(
            MODEL_PATH_V2,
            map_location="cpu",
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            classes = checkpoint.get(
                "classes"
            )

    # --------------------------------------------------------
    # Fallback to class names file
    # --------------------------------------------------------

    if classes is None:

        if os.path.exists(
            CLASS_NAMES_PATH_V2
        ):

            classes = torch.load(
                CLASS_NAMES_PATH_V2,
                map_location="cpu",
                weights_only=False
            )

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if classes is None:

        raise RuntimeError(
            "Could not load PlantAI V2 class names.\n"
            f"Model: {MODEL_PATH_V2}\n"
            f"Classes: {CLASS_NAMES_PATH_V2}"
        )

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    if not isinstance(
        classes,
        (list, tuple)
    ):

        raise RuntimeError(
            "PlantAI V2 class metadata is invalid."
        )

    classes = list(classes)

    if len(classes) != 38:

        raise RuntimeError(
            "PlantAI V2 class count mismatch. "
            f"Expected 38 classes, found {len(classes)}."
        )

    return classes


# ============================================================
# SAFE CLASS LABEL VALIDATION
# ============================================================

def validate_class_name(
    class_name,
    classes
):
    """
    Validate a user-supplied class name against the
    official PlantAI V2 38-class list.
    """

    if not class_name:

        raise ValueError(
            "Correct class name is required."
        )

    class_name = str(
        class_name
    ).strip()

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if class_name in classes:

        return class_name

    # --------------------------------------------------------
    # Case-insensitive match
    # --------------------------------------------------------

    lowered = class_name.lower()

    for known_class in classes:

        if known_class.lower() == lowered:

            return known_class

    # --------------------------------------------------------
    # Invalid label
    # --------------------------------------------------------

    raise ValueError(
        "Invalid class name. "
        "The supplied label does not exist "
        "in the PlantAI V2 38-class model."
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(
    image_path
):
    """
    Verify that the supplied file exists and is a valid image.
    """

    if not image_path:

        raise ValueError(
            "Image path is required."
        )

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image file does not exist: {image_path}"
        )

    extension = os.path.splitext(
        image_path
    )[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            "Unsupported image format. "
            "Supported formats: JPG, JPEG, PNG, WEBP."
        )

    try:

        with Image.open(
            image_path
        ) as image:

            image.verify()

    except Exception as e:

        raise ValueError(
            f"Invalid image file: {e}"
        )


# ============================================================
# FEEDBACK LOG
# ============================================================

def _write_log(
    original_filename,
    saved_filename,
    predicted_class,
    correct_class,
    confidence,
    feedback_type
):
    """
    Append a verified feedback event to feedback.csv.
    """

    initialize_learning_directories()

    file_exists = os.path.exists(
        FEEDBACK_LOG
    )

    with open(
        FEEDBACK_LOG,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        if not file_exists:

            writer.writerow([
                "timestamp",
                "original_filename",
                "saved_filename",
                "predicted_class",
                "correct_class",
                "confidence",
                "feedback_type"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            original_filename,
            saved_filename,
            predicted_class,
            correct_class,
            confidence,
            feedback_type
        ])


# ============================================================
# SAVE VERIFIED FEEDBACK
# ============================================================

def save_verified_feedback(
    image_path,
    correct_class,
    predicted_class="",
    confidence=0.0,
    feedback_type="user_verified"
):
    """
    Save a user-verified image into the appropriate
    verified/<class_name>/ directory.

    IMPORTANT:
        This function DOES NOT modify the production model.
    """

    initialize_learning_directories()

    # --------------------------------------------------------
    # Load official V2 classes
    # --------------------------------------------------------

    classes = get_model_classes()

    # --------------------------------------------------------
    # Validate corrected class
    # --------------------------------------------------------

    correct_class = validate_class_name(
        correct_class,
        classes
    )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    validate_image(
        image_path
    )

    # --------------------------------------------------------
    # Create class directory
    # --------------------------------------------------------

    class_directory = os.path.join(
        VERIFIED_DIR,
        correct_class
    )

    os.makedirs(
        class_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Determine extension
    # --------------------------------------------------------

    extension = os.path.splitext(
        image_path
    )[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:

        extension = ".jpg"

    # --------------------------------------------------------
    # Generate unique filename
    # --------------------------------------------------------

    unique_name = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:12]
        + extension
    )

    destination = os.path.join(
        class_directory,
        unique_name
    )

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        destination
    )

    # --------------------------------------------------------
    # Write feedback log
    # --------------------------------------------------------

    _write_log(
        os.path.basename(
            image_path
        ),
        unique_name,
        predicted_class,
        correct_class,
        confidence,
        feedback_type
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "success": True,
        "class": correct_class,
        "filename": unique_name,
        "path": destination
    }


# ============================================================
# COUNT VERIFIED SAMPLES
# ============================================================

def get_verified_counts():
    """
    Count verified images for every PlantAI V2 class.
    """

    initialize_learning_directories()

    classes = get_model_classes()

    counts = {}

    total = 0

    for class_name in classes:

        directory = os.path.join(
            VERIFIED_DIR,
            class_name
        )

        count = 0

        if os.path.isdir(
            directory
        ):

            for filename in os.listdir(
                directory
            ):

                path = os.path.join(
                    directory,
                    filename
                )

                if not os.path.isfile(
                    path
                ):

                    continue

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension in SUPPORTED_EXTENSIONS:

                    count += 1

        counts[class_name] = count

        total += count

    return {
        "total": total,
        "by_class": counts
    }


# ============================================================
# LEARNING STATUS
# ============================================================

def get_learning_status():
    """
    Return the current continuous-learning status.

    V2 remains the production model unless a V3 production
    model has explicitly been created.
    """

    initialize_learning_directories()

    counts = get_verified_counts()

    return {
        "learning_enabled": True,

        "verified_samples": counts["total"],

        "verified_by_class": counts["by_class"],

        "v3_candidate_exists": os.path.exists(
            MODEL_PATH_V3_CANDIDATE
        ),

        "v3_production_exists": os.path.exists(
            MODEL_PATH_V3
        ),

        "production_model": (
            "V3"
            if os.path.exists(MODEL_PATH_V3)
            else "V2"
        )
    }


# ============================================================
# INITIALIZE ON IMPORT
# ============================================================

initialize_learning_directories()


# ============================================================
# END OF FILE
# ============================================================