import torch

from torchvision import datasets, transforms

from torch.utils.data import (
    DataLoader,
    Subset
)

from config import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE
)


# ============================================================
# ImageNet normalization
# ============================================================

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        20
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    # --------------------------------------------------------
    # Load dataset once to obtain class names
    # --------------------------------------------------------

    full_dataset = datasets.ImageFolder(
        root=DATASET_PATH
    )


    total_size = len(
        full_dataset
    )


    classes = full_dataset.classes


    num_classes = len(
        classes
    )


    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if total_size == 0:

        raise RuntimeError(
            f"No images found in dataset: {DATASET_PATH}"
        )


    if num_classes < 2:

        raise RuntimeError(
            "Dataset must contain at least 2 classes."
        )


    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_size = int(
        0.8 * total_size
    )

    val_size = (
        total_size
        - train_size
    )


    generator = torch.Generator()

    generator.manual_seed(
        42
    )


    train_indices, val_indices = (
        torch.utils.data.random_split(

            range(total_size),

            [
                train_size,
                val_size
            ],

            generator=generator
        )
    )


    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset_full = datasets.ImageFolder(

        root=DATASET_PATH,

        transform=train_transform

    )


    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset_full = datasets.ImageFolder(

        root=DATASET_PATH,

        transform=val_transform

    )


    # --------------------------------------------------------
    # Subsets
    # --------------------------------------------------------

    train_dataset = Subset(

        train_dataset_full,

        train_indices.indices

    )


    val_dataset = Subset(

        val_dataset_full,

        val_indices.indices

    )


    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=torch.cuda.is_available()

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()

    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "🌱 PLANTAI DATASET"
    )

    print(
        "=" * 60
    )

    print(
        f"Dataset path : {DATASET_PATH}"
    )

    print(
        f"Classes      : {num_classes}"
    )

    print(
        f"Images       : {total_size}"
    )

    print(
        f"Training     : {train_size}"
    )

    print(
        f"Validation   : {val_size}"
    )

    print()

    print(
        "Classes:"
    )


    for index, class_name in enumerate(
        classes
    ):

        print(
            f"{index:02d} : {class_name}"
        )


    print(
        "=" * 60
    )

    print()


    return (
        train_loader,
        val_loader,
        classes
    )