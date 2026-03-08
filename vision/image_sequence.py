import cv2
import os
import random

def read_sequence(folder_path, max_frames=3000, random_start=True):
    # List only image files
    images = sorted([
        img for img in os.listdir(folder_path)
        if img.lower().endswith(".jpg")
    ])

    if len(images) == 0:
        raise ValueError("No images found in the given folder")

    # Choose random start index
    if random_start:
        max_start = max(0, len(images) - max_frames)
        start_idx = random.randint(0, max_start)
    else:
        start_idx = 0

    end_idx = start_idx + max_frames

    print(f"[INFO] Starting from frame index: {start_idx}")

    for img in images[start_idx:end_idx]:
        img_path = os.path.join(folder_path, img)
        frame = cv2.imread(img_path)

        if frame is not None:
            yield frame
