import cv2
import os

# Input video
video_path = "/home/arslan-sadiq/Pictures/gree/video_40min_to_57min.mp4"

# Output folder
output_folder = "/home/arslan-sadiq/Pictures/gree/ac-data"

os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("Error: Could not determine video FPS.")
    cap.release()
    exit()

print(f"Video FPS: {fps}")

frame_number = 0
saved_frame = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save one frame every second
    if frame_number % round(fps) == 0:
        output_path = os.path.join(
            output_folder,
            f"frame_{saved_frame:05d}.jpg"
        )

        cv2.imwrite(output_path, frame)
        saved_frame += 1

    frame_number += 1

cap.release()

print(f"Done! {saved_frame} frames saved in '{output_folder}'.")
