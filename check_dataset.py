from pathlib import Path
from torchvision import datasets

DATASET_PATH = Path("dataset_global")

dataset = datasets.ImageFolder(DATASET_PATH)

print("=" * 70)
print("PLANTAI DATASET CHECK")
print("=" * 70)

print("Total images :", len(dataset))
print("Total classes:", len(dataset.classes))

print()
print("CLASS DISTRIBUTION")
print("=" * 70)

counts = [0] * len(dataset.classes)

for _, class_index in dataset.samples:
    counts[class_index] += 1

for i, (class_name, count) in enumerate(
    zip(dataset.classes, counts)
):

    print(
        f"{i:02d} | {class_name:60s} | {count}"
    )

print()
print("=" * 70)