"""
YOLO Inference Script - Sorts frames into 'detected' and 'no_detection' folders
based on whether the model finds any objects in each image.

Just edit the CONFIG section below with your paths and run:
    python infer_and_sort.py

Requirements:
    pip install ultralytics
"""

import shutil
from pathlib import Path
from ultralytics import YOLO


# ============================ CONFIG - EDIT THESE ============================
MODEL_PATH = "/home/arslan-sadiq/Pictures/TATA/18-SEP/TATA-CAM2.pt"         # path to your trained model file
SOURCE_DIR = "/home/arslan-sadiq/Pictures/TATA/18-SEP/Tata-Cam2-cleaned-1"           # folder containing input frames/images
OUTPUT_DIR = "/home/arslan-sadiq/Pictures/TATA/18-SEP/Cam2-Cleaned"           # folder where sorted results will be saved

CONF_THRESHOLD = 0.50                    # confidence threshold
IOU_THRESHOLD = 0.45                     # IoU threshold for NMS
CLASSES = None                           # e.g. [0, 2] to filter specific class IDs, or None for all
DEVICE = None                            # e.g. "cpu", "0", "0,1", or None for auto

SAVE_LABELS = False                      # also save YOLO .txt label files for detected frames
SAVE_ANNOTATED = False                   # save images with boxes drawn instead of raw originals in 'detected'
# ==============================================================================


def main():
    model_path = Path(MODEL_PATH)
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {source_path}")

    detected_dir = output_path / "detected"
    no_detection_dir = output_path / "no_detection"
    labels_dir = output_path / "detected_labels"

    detected_dir.mkdir(parents=True, exist_ok=True)
    no_detection_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_LABELS:
        labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = sorted([f for f in source_path.iterdir() if f.suffix.lower() in valid_exts])

    if not image_files:
        print(f"No image files found in {source_path}")
        return

    print(f"Found {len(image_files)} frames. Running inference...\n")

    detected_count = 0
    no_detection_count = 0

    for idx, img_path in enumerate(image_files, 1):
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=CLASSES,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        has_detections = result.boxes is not None and len(result.boxes) > 0

        if has_detections:
            detected_count += 1
            dest = detected_dir / img_path.name

            if SAVE_ANNOTATED:
                annotated = result.plot()  # numpy array (BGR)
                import cv2
                cv2.imwrite(str(dest), annotated)
            else:
                shutil.copy2(img_path, dest)

            if SAVE_LABELS:
                label_path = labels_dir / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        x, y, w, h = box.xywhn[0].tolist()  # normalized YOLO format
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            status = f"{len(result.boxes)} object(s)"
        else:
            no_detection_count += 1
            dest = no_detection_dir / img_path.name
            shutil.copy2(img_path, dest)
            status = "no detection"

        print(f"[{idx}/{len(image_files)}] {img_path.name} -> {status}")

    print("\n--- Summary ---")
    print(f"Total frames processed : {len(image_files)}")
    print(f"Detected               : {detected_count}  -> {detected_dir}")
    print(f"No detection           : {no_detection_count}  -> {no_detection_dir}")
    if SAVE_LABELS:
        print(f"Labels saved to        : {labels_dir}")


if __name__ == "__main__":
    main()