import shutil
import hashlib
from pathlib import Path

from PIL import Image
import imagehash


# ============================================================
# CONFIGURATION
# ============================================================

# Your ORIGINAL YOLO dataset.
#
# Expected source structure:
#
# TATACAM1-Final/
# ├── images/
# │   ├── image001.jpg
# │   ├── image002.jpg
# │   └── ...
# │
# └── labels/
#     ├── image001.txt
#     ├── image002.txt
#     └── ...
#
SOURCE_DIR = Path(
    "/home/arslan-sadiq/Downloads/Image_Cleaner/TATACAM1-Final"
)

# Clean dataset will be created here.
OUTPUT_DIR = SOURCE_DIR.parent / "no-duplicate"

OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_LABELS_DIR = OUTPUT_DIR / "labels"

# pHash threshold
# 0     = very strict
# 1-3   = conservative
# 4-5   = good starting point
# 6-8   = more aggressive
PHASH_THRESHOLD = 8

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
}


# ============================================================
# SHA256 - EXACT DUPLICATE
# ============================================================

def get_sha256(image_path):
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        while True:
            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# pHASH - NEAR DUPLICATE
# ============================================================

def get_phash(image_path):
    try:
        with Image.open(image_path) as image:
            return imagehash.phash(image)

    except Exception as error:
        print()
        print("[ERROR] Could not calculate pHash:")
        print(f"        {image_path}")
        print(f"        {error}")
        return None


# ============================================================
# FIND IMAGES
# ============================================================

def get_images():
    """
    Find images ONLY inside the source images/ directory.

    This is important for YOLO datasets where images and labels
    are stored in separate directories.
    """

    source_images_dir = SOURCE_DIR / "images"

    if not source_images_dir.exists():
        print("[ERROR] Source images directory does not exist:")
        print(source_images_dir)
        print()
        print("Expected structure:")
        print("TATACAM1-Final/")
        print("├── images/")
        print("└── labels/")
        return []

    images = []

    for path in source_images_dir.rglob("*"):

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


# ============================================================
# FIND CORRESPONDING YOLO LABEL
# ============================================================

def get_label_path(image_path):
    """
    Convert:

        SOURCE_DIR/images/subfolder/image001.jpg

    into:

        SOURCE_DIR/labels/subfolder/image001.txt
    """

    source_images_dir = SOURCE_DIR / "images"
    source_labels_dir = SOURCE_DIR / "labels"

    relative_image_path = image_path.relative_to(source_images_dir)

    label_relative_path = relative_image_path.with_suffix(".txt")

    label_path = source_labels_dir / label_relative_path

    return label_path


# ============================================================
# COPY UNIQUE IMAGE + LABEL
# ============================================================

def copy_unique_image_and_label(image_path):
    """
    Copy the image to:

        no-duplicate/images/

    Copy its corresponding YOLO label to:

        no-duplicate/labels/
    """

    source_images_dir = SOURCE_DIR / "images"

    relative_path = image_path.relative_to(source_images_dir)

    # -----------------------------
    # Copy image
    # -----------------------------

    output_image_path = OUTPUT_IMAGES_DIR / relative_path

    output_image_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        image_path,
        output_image_path
    )

    # -----------------------------
    # Find corresponding label
    # -----------------------------

    label_path = get_label_path(image_path)

    if label_path.exists():

        output_label_path = (
            OUTPUT_LABELS_DIR
            / relative_path.with_suffix(".txt")
        )

        output_label_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            label_path,
            output_label_path
        )

        return True

    else:

        print()
        print("[WARNING] Label not found")
        print(f"Image : {image_path}")
        print(f"Expected label: {label_path}")

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO IMAGE NEAR-DUPLICATE CLEANER")
    print("=" * 70)

    print(f"Original dataset : {SOURCE_DIR}")
    print(f"Clean dataset    : {OUTPUT_DIR}")
    print(f"pHash threshold  : {PHASH_THRESHOLD}")
    print()

    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():
        print("[ERROR] Source dataset does not exist:")
        print(SOURCE_DIR)
        return

    source_images_dir = SOURCE_DIR / "images"
    source_labels_dir = SOURCE_DIR / "labels"

    if not source_images_dir.exists():
        print("[ERROR] Missing:")
        print(source_images_dir)
        return

    if not source_labels_dir.exists():
        print("[ERROR] Missing:")
        print(source_labels_dir)
        return

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUTPUT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    images = get_images()

    print(f"Images found: {len(images)}")
    print()

    if not images:
        print("No images found.")
        return

    # --------------------------------------------------------
    # STEP 1: EXACT DUPLICATES
    # --------------------------------------------------------

    print("=" * 70)
    print("STEP 1: EXACT DUPLICATE CHECK - SHA256")
    print("=" * 70)

    sha256_map = {}

    exact_duplicates = []
    remaining_images = []

    for image_path in images:

        file_hash = get_sha256(image_path)

        if file_hash in sha256_map:

            original = sha256_map[file_hash]

            print()
            print("[EXACT DUPLICATE]")
            print(f"Original : {original.name}")
            print(f"Duplicate: {image_path.name}")
            print("Action   : Not copied")

            exact_duplicates.append(image_path)

        else:

            sha256_map[file_hash] = image_path
            remaining_images.append(image_path)

    print()
    print(f"Exact duplicates found: {len(exact_duplicates)}")

    # --------------------------------------------------------
    # STEP 2: NEAR DUPLICATES - pHASH
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2: NEAR-DUPLICATE CHECK - pHASH")
    print("=" * 70)

    phash_database = []

    near_duplicates = []
    unique_images = []

    for index, image_path in enumerate(
        remaining_images,
        start=1
    ):

        print(
            f"\rProcessing {index}/{len(remaining_images)}",
            end="",
            flush=True
        )

        current_hash = get_phash(image_path)

        if current_hash is None:
            continue

        duplicate_found = False

        for original_path, original_hash in phash_database:

            distance = current_hash - original_hash

            if distance <= PHASH_THRESHOLD:

                print()
                print()
                print("[NEAR DUPLICATE]")
                print(f"Original       : {original_path.name}")
                print(f"Duplicate      : {image_path.name}")
                print(f"pHash distance : {distance}")
                print("Action         : Not copied")

                near_duplicates.append(image_path)

                duplicate_found = True
                break

        if not duplicate_found:

            phash_database.append(
                (image_path, current_hash)
            )

            unique_images.append(image_path)

    print()

    # --------------------------------------------------------
    # STEP 3: COPY UNIQUE IMAGES + LABELS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 3: COPYING UNIQUE IMAGES + YOLO LABELS")
    print("=" * 70)

    copied_images = 0
    copied_labels = 0
    missing_labels = 0

    for image_path in unique_images:

        label_copied = copy_unique_image_and_label(
            image_path
        )

        copied_images += 1

        if label_copied:
            copied_labels += 1
        else:
            missing_labels += 1

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_duplicates = (
        len(exact_duplicates)
        + len(near_duplicates)
    )

    print()
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(f"Original images       : {len(images)}")
    print(f"Exact duplicates      : {len(exact_duplicates)}")
    print(f"Near duplicates       : {len(near_duplicates)}")
    print(f"Total duplicates      : {total_duplicates}")
    print(f"Unique images         : {len(unique_images)}")
    print()
    print(f"Images copied         : {copied_images}")
    print(f"Labels copied         : {copied_labels}")
    print(f"Missing labels        : {missing_labels}")
    print()
    print("CLEAN DATASET")
    print("-" * 70)
    print(f"Images: {OUTPUT_IMAGES_DIR}")
    print(f"Labels: {OUTPUT_LABELS_DIR}")
    print()
    print("Original dataset was NOT modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
