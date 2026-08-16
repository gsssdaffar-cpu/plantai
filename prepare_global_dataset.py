import os
import shutil

SOURCE_DIR = os.path.join(
    "PlantVillage_Source",
    "raw",
    "color"
)

OUTPUT_DIR = "dataset_global"


def main():

    print("=" * 60)
    print("PlantAI - PlantVillage Dataset Preparation")
    print("=" * 60)

    if not os.path.exists(SOURCE_DIR):
        print("ERROR: PlantVillage source folder not found:")
        print(SOURCE_DIR)
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    class_folders = [
        folder
        for folder in os.listdir(SOURCE_DIR)
        if os.path.isdir(
            os.path.join(SOURCE_DIR, folder)
        )
    ]

    class_folders.sort()

    print()
    print(f"Classes found: {len(class_folders)}")
    print()

    total_images = 0

    for class_name in class_folders:

        source_class_path = os.path.join(
            SOURCE_DIR,
            class_name
        )

        output_class_path = os.path.join(
            OUTPUT_DIR,
            class_name
        )

        os.makedirs(
            output_class_path,
            exist_ok=True
        )

        class_count = 0

        # Search recursively because this PlantVillage
        # version stores images inside UUID folders.
        for root, dirs, files in os.walk(
            source_class_path
        ):

            for filename in files:

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                ]:
                    continue

                source_file = os.path.join(
                    root,
                    filename
                )

                # Give every image a unique filename.
                new_filename = (
                    f"{class_count:06d}"
                    + extension
                )

                destination_file = os.path.join(
                    output_class_path,
                    new_filename
                )

                # Copy, don't move.
                shutil.copy2(
                    source_file,
                    destination_file
                )

                class_count += 1
                total_images += 1

        print(
            f"{class_name:<50} {class_count:>5} images"
        )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Classes : {len(class_folders)}")
    print(f"Images  : {total_images}")
    print(f"Output  : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()