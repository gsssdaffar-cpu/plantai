import os

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    CLASS_NAMES_PATH
)

from dataset import load_dataset
from model import create_model
from engine import train_one_epoch, validate


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

train_loader, val_loader, classes = load_dataset()


num_classes = len(classes)


print(
    f"Number of classes: {num_classes}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = create_model(
    num_classes,
    pretrained=True
)


model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# BEST ACCURACY
# ============================================================

best_accuracy = 0.0


# ============================================================
# TRAIN
# ============================================================

for epoch in range(EPOCHS):

    print()
    print(
        "=" * 60
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_acc = train_one_epoch(

        model,

        train_loader,

        criterion,

        optimizer,

        DEVICE

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_acc = validate(

        model,

        val_loader,

        criterion,

        DEVICE

    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()

    print(
        f"Train Loss       : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : {train_acc:.2f}%"
    )

    print(
        f"Validation Loss  : {val_loss:.4f}"
    )

    print(
        f"Validation Acc   : {val_acc:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_acc > best_accuracy:

        best_accuracy = val_acc


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "classes":
                classes,

            "num_classes":
                num_classes,

            "image_size":
                224,

            "best_accuracy":
                best_accuracy

        }


        torch.save(

            checkpoint,

            MODEL_PATH

        )


        # Also save class names separately

        torch.save(

            classes,

            CLASS_NAMES_PATH

        )


        print()
        print(
            "✅ Best model saved."
        )

        print(
            f"Validation Accuracy: "
            f"{best_accuracy:.2f}%"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "=" * 60
)

print(
    "🌱 PLANTAI TRAINING COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Classes: {CLASS_NAMES_PATH}"
)

print(
    "=" * 60
)