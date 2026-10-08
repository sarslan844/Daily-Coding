import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

# Define each stream with its own model_path, name, url, and output_dir.
STREAMS = [
    {
        "name": "TataCam1",
        "url": "https://tata1-stream.betacodespk.com/raw/TataCam1/live.m3u8",
        "output_dir": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/TataCam1",
        "model_path": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/BEST mODELS/TATA-CAM1.pt",
    },
    {
        "name": "TataCam2",
        "url": "https://tata1-stream.betacodespk.com/raw/TataCam2/live.m3u8",
        "output_dir": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/TataCam2",
        "model_path": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/BEST mODELS/Tata_cam2_best.pt",
    },
    {
        "name": "Masood",
        "url": "https://masood-textile.betacodespk.com/raw/masoodCam2/live.m3u8",
        "output_dir": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/Masood",
        "model_path": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/BEST mODELS/massod-best.pt",
    },
    {
        "name": "Artic",
        "url": "https://arctic1-stream.betacodespk.com/raw/arcticCam2/live.m3u8",
        "output_dir": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/Artic",
        "model_path": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/BEST mODELS/Artic1.pt",
    },
    # Add more streams like this:
    # {
    #     "name": "TataCam3",
    #     "url": "https://.../live.m3u8",
    #     "output_dir": "/home/arslan-sadiq/Downloads/rtsp_yolo_annotator/TataCam3-final",
    #     "model_path": "/path/to/Tata_cam3_best.pt",
    # },
]

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
FRAME_BYTES = STREAM_WIDTH * STREAM_HEIGHT * 3

CONFIDENCE_THRESHOLD = 0.30
MIN_BOX_AREA = 400
CLASS_NAMES = ["cotton_bale"]

# 0.5 = one frame every 2 seconds
TARGET_FPS = 0.5

# Reconnect backoff: starts at RECONNECT_DELAY, doubles each failure,
# capped at RECONNECT_DELAY_MAX, and resets after a successful frame.
RECONNECT_DELAY = 2
RECONNECT_DELAY_MAX = 60

# Timeout in seconds to wait for a frame from FFmpeg before reconnecting.
# Increased to 15 to accommodate HLS (.m3u8) buffer/download latency.
FRAME_TIMEOUT = 15

# Log file (rotates at 5 MB, keeps 3 backups). Set to None to log to console only.
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.log")

# Matches saved files like image_000123.jpg / image_000123.txt
FILE_NUMBER_RE = re.compile(r"^image_(\d{6})\.(jpg|txt)$")

# ============================================================
# GLOBAL STATE
# ============================================================

running = True
log = logging.getLogger("SYSTEM")


def setup_logging():
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if LOG_FILE:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def shutdown_handler(signum, frame):
    global running
    log.info("Shutdown signal received. Stopping all streams...")
    running = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_last_counter(image_dir, label_dir):
    """
    Return the highest image number already used in image_dir/label_dir,
    so a restart continues numbering instead of overwriting old data.
    Also removes leftover temp files from a previous crash.
    """
    highest = 0
    for directory in (image_dir, label_dir):
        for filename in os.listdir(directory):
            if filename.endswith(".tmp") or ".tmp." in filename:
                try:
                    os.remove(os.path.join(directory, filename))
                except OSError:
                    pass
                continue
            match = FILE_NUMBER_RE.match(filename)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest


def save_yolo_label_atomic(label_path, detections, img_w, img_h):
    """Write label to a temp file, then atomically rename it into place."""
    tmp_path = label_path + ".tmp"
    with open(tmp_path, "w") as f:
        for det in detections:
            cid = det["class_id"]
            x1, y1, x2, y2 = det["bbox"]

            center_x = np.clip(((x1 + x2) / 2.0) / img_w, 0.0, 1.0)
            center_y = np.clip(((y1 + y2) / 2.0) / img_h, 0.0, 1.0)
            width = np.clip((x2 - x1) / img_w, 0.0, 1.0)
            height = np.clip((y2 - y1) / img_h, 0.0, 1.0)

            f.write(
                f"{cid} {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )
    os.replace(tmp_path, label_path)


def save_image_atomic(image_path, frame):
    """Write image to a temp file (keeps .jpg extension for OpenCV), then rename."""
    base, ext = os.path.splitext(image_path)
    tmp_path = f"{base}.tmp{ext}"
    if not cv2.imwrite(tmp_path, frame):
        return False
    os.replace(tmp_path, image_path)
    return True


def save_pair_atomic(image_path, label_path, frame, detections):
    """
    Label is written first, image last. A saved image therefore always has
    its label; a crash can at worst leave an orphan label, which is
    harmless (its number is skipped on restart).
    """
    save_yolo_label_atomic(label_path, detections, STREAM_WIDTH, STREAM_HEIGHT)
    if not save_image_atomic(image_path, frame):
        try:
            os.remove(label_path)
        except OSError:
            pass
        return False
    return True


def start_ffmpeg_process(stream_url):
    """
    Start one independent FFmpeg process for one stream.
    Optimized for HLS live stream stability.
    """
    command = [
        "ffmpeg",
        "-loglevel", "warning",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",

        # Fix timestamp issues and tolerate corrupt chunks in HLS live streams
        "-fflags", "+genpts+discardcorrupt",

        "-i", stream_url,

        "-map", "0:v:0",
        "-an",

        "-vf", f"fps={TARGET_FPS},scale={STREAM_WIDTH}:{STREAM_HEIGHT}",

        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )


def read_exact(pipe, buffer):
    """
    Fill `buffer` completely from `pipe`.
    A single read() on a pipe may return fewer bytes than requested,
    so we loop until the whole frame has arrived.
    Returns False only if the stream really ended (EOF).
    """
    view = memoryview(buffer)
    total = len(buffer)
    received = 0

    while received < total:
        n = pipe.readinto(view[received:])
        if not n:  # 0 or None -> EOF
            return False
        received += n
    return True


def frame_buffer_worker(process, frame_queue, stream_name):
    """
    Reads FFmpeg frames continuously.
    Queue size is 1 so old frames are discarded and the YOLO
    worker always receives the freshest available frame.
    """
    logger = logging.getLogger(stream_name)

    while running and process.poll() is None:
        # New buffer per frame: a queued frame is never overwritten.
        buffer = bytearray(FRAME_BYTES)

        try:
            if not read_exact(process.stdout, buffer):
                break
        except Exception as e:
            logger.error("FFmpeg read error: %s", e)
            break

        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(
            (STREAM_HEIGHT, STREAM_WIDTH, 3)
        )

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass


def process_detections(frame, model, target_class_ids, device):
    """
    Run the stream's custom YOLO model on a frame and return matching detections.
    """
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        if class_id not in target_class_ids:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        detections.append({
            "class_id": class_id,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
        })

    return detections


def interruptible_sleep(seconds):
    """Sleep in small steps so Ctrl+C / SIGTERM stops us quickly."""
    end = time.monotonic() + seconds
    while running and time.monotonic() < end:
        time.sleep(0.2)


# ============================================================
# PER-STREAM WORKER
# ============================================================

def stream_worker(stream, device):
    """
    Complete processing pipeline for one stream.
    Every stream loads its OWN YOLO model independently.
    """
    stream_name = stream["name"]
    stream_url = stream["url"]
    model_path = stream["model_path"]
    logger = logging.getLogger(stream_name)

    # 1. Load the dedicated model for this specific stream
    logger.info("Loading stream-specific model: %s", model_path)
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error("Could not load model: %s", e)
        return

    # 2. Extract target class IDs for this specific model
    target_class_ids = [
        int(cid)
        for cid, name in model.names.items()
        if name in CLASS_NAMES
    ]

    if not target_class_ids:
        logger.error(
            "Target classes %s not found in model classes: %s",
            CLASS_NAMES, model.names,
        )
        return

    logger.info(
        "Model loaded. Classes: %s | Target IDs: %s", model.names, target_class_ids
    )

    frame_queue = queue.Queue(maxsize=1)

    # Resume numbering so restarts never overwrite existing data.
    image_counter = get_last_counter(stream["image_dir"], stream["label_dir"])
    if image_counter:
        logger.info("Resuming from image number %d", image_counter + 1)

    reconnect_delay = RECONNECT_DELAY
    logger.info("Worker started.")

    while running:
        ffmpeg_proc = None
        reader_thread = None
        got_frame = False  # did this connection deliver at least one frame?

        try:
            logger.info("Starting FFmpeg...")

            ffmpeg_proc = start_ffmpeg_process(stream_url)

            reader_thread = threading.Thread(
                target=frame_buffer_worker,
                args=(ffmpeg_proc, frame_queue, stream_name),
                daemon=True,
            )
            reader_thread.start()

            logger.info("Stream connected.")

            while running and ffmpeg_proc.poll() is None:

                try:
                    frame = frame_queue.get(timeout=FRAME_TIMEOUT)
                except queue.Empty:
                    logger.warning(
                        "Frame timeout (%ss). Reconnecting...", FRAME_TIMEOUT
                    )
                    break

                # A frame arrived -> connection is healthy, reset backoff.
                got_frame = True
                reconnect_delay = RECONNECT_DELAY

                detections = process_detections(
                    frame, model, target_class_ids, device
                )

                if not detections:
                    continue

                next_number = image_counter + 1
                img_filename = f"image_{next_number:06d}.jpg"
                lbl_filename = f"image_{next_number:06d}.txt"

                image_path = os.path.join(stream["image_dir"], img_filename)
                label_path = os.path.join(stream["label_dir"], lbl_filename)

                try:
                    saved = save_pair_atomic(image_path, label_path, frame, detections)
                except OSError as e:
                    logger.error("Disk write failed: %s", e)
                    continue

                if not saved:
                    logger.error("Failed to save image: %s", image_path)
                    continue

                image_counter = next_number
                logger.info("[SAVED] %s | Detections: %d", img_filename, len(detections))

        except Exception:
            logger.exception("Worker exception")

        finally:
            if ffmpeg_proc is not None:
                try:
                    if ffmpeg_proc.poll() is None:
                        ffmpeg_proc.kill()
                    ffmpeg_proc.wait(timeout=3)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                    except Exception:
                        pass

        if running:
            logger.info("Waiting %ss before reconnect...", reconnect_delay)
            interruptible_sleep(reconnect_delay)
            # Exponential backoff; stays reset if the last connection was healthy.
            if got_frame:
                reconnect_delay = RECONNECT_DELAY
            else:
                reconnect_delay = min(reconnect_delay * 2, RECONNECT_DELAY_MAX)

    logger.info("Worker stopped. Last image number: %d", image_counter)


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Inference device: %s", device)
    log.info("Total number of streams: %d", len(STREAMS))

    # Path verification & output directory setup
    for stream in STREAMS:
        if not os.path.exists(stream["model_path"]):
            log.critical("Model file not found for %s: %s",
                         stream["name"], stream["model_path"])
            log.critical("Please check for case-sensitivity or typos in the file path.")
            sys.exit(1)

        stream["image_dir"] = os.path.join(stream["output_dir"], "images")
        stream["label_dir"] = os.path.join(stream["output_dir"], "labels")

        os.makedirs(stream["image_dir"], exist_ok=True)
        os.makedirs(stream["label_dir"], exist_ok=True)

    log.info("=" * 70)
    log.info("Starting MULTI-STREAM YOLO pipeline")
    log.info("=" * 70)

    workers = []
    for stream in STREAMS:
        worker = threading.Thread(
            target=stream_worker,
            args=(stream, device),
            daemon=True,
        )
        workers.append(worker)
        worker.start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        global_stop()

    log.info("Waiting for all stream workers to stop...")
    for worker in workers:
        worker.join(timeout=5)

    log.info("All streams stopped safely.")


def global_stop():
    global running
    running = False


if __name__ == "__main__":
    main()