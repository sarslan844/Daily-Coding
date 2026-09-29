import cv2
import os


# ==========================================
# SETTINGS
# ==========================================

# Change this to your video path
input_video = "/home/arslan-sadiq/Pictures/gree/video/v19.mp4"

# Output folder
output_folder = "/home/arslan-sadiq/Pictures/gree/ac-data"

# Output video filename
output_video = os.path.join(
    output_folder,
    "video_40min_to_57min.mp4"
)

# Cut times
start_time = 40 * 60   # 40 minutes = 2400 seconds
end_time = 57 * 60     # 57 minutes = 3420 seconds


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(output_folder, exist_ok=True)


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    exit()


# ==========================================
# GET VIDEO INFORMATION
# ==========================================

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / fps


print("----------------------------------------")
print("Video Information")
print("----------------------------------------")
print(f"FPS: {fps}")
print(f"Resolution: {width} x {height}")
print(f"Duration: {duration:.2f} seconds")
print("Cutting: 40:00 -> 57:00")
print("----------------------------------------")


# ==========================================
# CHECK VIDEO LENGTH
# ==========================================

if duration < end_time:
    print("ERROR: Video is shorter than 57 minutes.")
    cap.release()
    exit()


# ==========================================
# MOVE TO 40 MINUTES
# ==========================================

cap.set(
    cv2.CAP_PROP_POS_MSEC,
    start_time * 1000
)


# ==========================================
# CREATE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()


# ==========================================
# CUT VIDEO
# ==========================================

print("Starting video extraction...")

frame_count = 0

while True:

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Stop at 57 minutes
    if current_time >= end_time:
        break

    ret, frame = cap.read()

    if not ret:
        break

    out.write(frame)

    frame_count += 1

    # Show progress every 100 frames
    if frame_count % 100 == 0:
        elapsed = current_time - start_time

        print(
            f"\rProcessing: {elapsed / 60:.2f} minutes",
            end="",
            flush=True
        )


# ==========================================
# RELEASE
# ==========================================

cap.release()
out.release()


# ==========================================
# DONE
# ==========================================

print("\n")
print("----------------------------------------")
print("Video cutting completed successfully!")
print("----------------------------------------")
print(f"Output file:")
print(output_video)
print("----------------------------------------")