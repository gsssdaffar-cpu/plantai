import torch


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = "dataset_global"


# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 16

# Start with 3 for testing.
# After confirming everything works, use 15-30.
EPOCHS = 20

# Learning rate for fine-tuning pretrained EfficientNet
LEARNING_RATE = 0.0001


# ============================================================
# VALIDATION
# ============================================================

TRAIN_SPLIT = 0.80


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/plant_model_38class.pth"

CLASS_NAMES_PATH = "models/class_names.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Using device: {DEVICE}"
)


if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    print(
        "GPU not available. Training will use CPU."
    )