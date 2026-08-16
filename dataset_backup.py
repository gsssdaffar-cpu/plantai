from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

from config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE


# ============================================================
# TRAINING TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        20
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION TRANSFORMS
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    # First dataset used only to obtain file paths/classes
    base_dataset = datasets.ImageFolder(
        root=DATASET_PATH
    )

    train_size = int(
        0.8 * len(base_dataset)
    )

    val_size = (
        len(base_dataset) - train_size
    )


    # Reproducible split
    generator = __import__("torch").Generator()

    generator.manual_seed(42)


    train_indices, val_indices = random_split(
        range(len(base_dataset)),
        [train_size, val_size],
        generator=generator
    )


    # Training dataset
    train_full = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=train_transform
    )


    # Validation dataset
    val_full = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=val_transform
    )


    train_dataset = __import__(
        "torch.utils.data",
        fromlist=["Subset"]
    ).Subset(
        train_full,
        train_indices.indices
    )


    val_dataset = __import__(
        "torch.utils.data",
        fromlist=["Subset"]
    ).Subset(
        val_full,
        val_indices.indices
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    return (
        train_loader,
        val_loader,
        base_dataset.classes
    )