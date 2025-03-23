import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("./data/normalized_landmarks.csv")

POSE_CONNECTIONS = [
    # Arms
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    # Shoulders and hips
    (11, 12), (23, 24), (11, 23), (12, 24),
    # Left leg
    (23, 25), (25, 27), (27, 31), (31, 29),
    # Right leg
    (24, 26), (26, 28), (28, 32), (32, 30)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),   # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def extract_landmarks(row, prefix, num_landmarks):
    landmarks = []
    for i in range(num_landmarks):
        x = row.get(f"{prefix}_{i}_x", np.nan)
        y = row.get(f"{prefix}_{i}_y", np.nan)
        z = row.get(f"{prefix}_{i}_z", np.nan)
        landmarks.append([x, y, z])
    arr = np.array(landmarks)
    if prefix == "right_hand":
        print(f"Right Hand Landmarks (First 3): {arr[:3]}")  # Debug
    return arr


def scale_all_landmarks(pose, left_hand, right_hand, face, width=1280, height=720):
    # Define the strict processing order
    processing_order = ['pose', 'left_hand', 'right_hand', 'face']
    parts = {
        'pose': pose,
        'left_hand': left_hand,
        'right_hand': right_hand,
        'face': face
    }

    # Collect valid parts in the defined order
    valid = []
    counts = []
    for key in processing_order:
        arr = parts[key]
        if arr is not None and arr.size > 0:
            valid.append(key)
            counts.append(arr.shape[0])

    if not valid:
        return pose, left_hand, right_hand, face

    # Combine and scale landmarks (same as before)
    combined = np.concatenate([
        pose, left_hand, right_hand, face
    ], axis=0) if any([pose.size, left_hand.size, right_hand.size, face.size]) else np.array([])
    min_x, max_x = np.nanmin(combined[:, 0]), np.nanmax(combined[:, 0])
    min_y, max_y = np.nanmin(combined[:, 1]), np.nanmax(combined[:, 1])
    x_range = max(max_x - min_x, 0.001)
    y_range = max(max_y - min_y, 0.001)

    for i in range(len(combined)):
        x, y, _ = combined[i]
        if not (np.isnan(x) or np.isnan(y)):
            screen_x = ((x - min_x) / x_range) * width * 0.9 + width * 0.05
            screen_y = ((y - min_y) / y_range) * height * 0.9 + height * 0.05
            combined[i] = [screen_x, screen_y, 0]

    # Split and assign back in strict order
    split_arrays = []
    idx = 0
    for count in counts:
        split_arrays.append(combined[idx:idx + count])
        idx += count

    # Reassign using the original processing order
    new_parts = {'pose': None, 'left_hand': None, 'right_hand': None, 'face': None}
    for key, arr in zip(valid, split_arrays):
        new_parts[key] = arr

    return (
        new_parts['pose'],
        new_parts['left_hand'],
        new_parts['right_hand'],
        new_parts['face']
    )

def draw_landmarks(img, landmarks, color):
    if landmarks is None or landmarks.size == 0:
        return
    for x, y, _ in landmarks:
        if not (np.isnan(x) or np.isnan(y)):
            x, y = round(x), round(y)
            cv2.circle(img, (x, y), 5, color, -1)

def draw_skeleton(img, landmarks, connections, color):
    if landmarks is None or landmarks.size == 0:
        return
    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        x1, y1, _ = landmarks[start_idx]
        x2, y2, _ = landmarks[end_idx]
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        h, w, _ = img.shape
        if (0 <= x1 < w and 0 <= y1 < h) and (0 <= x2 < w and 0 <= y2 < h):
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

def interpolate_landmarks(prev, new, alpha):
    if prev is None or prev.size == 0 or prev.shape != new.shape:
        return new
    # If new data is all-NaN, retain previous data
    if np.isnan(new).all():
        return prev
    # Blend only valid points
    mask = ~np.isnan(new)
    interpolated = prev.copy()
    interpolated[mask] = prev[mask] * (1 - alpha) + new[mask] * alpha
    return interpolated

image_size = (720, 1280, 3)
prev_frame = None

for i in range(len(df)):
    img = np.ones(image_size, dtype=np.uint8) * 255
    row = df.iloc[i]

    pose_landmarks = extract_landmarks(row, "pose", 33)
    left_hand_landmarks = extract_landmarks(row, "left_hand", 21)
    right_hand_landmarks = extract_landmarks(row, "right_hand", 21)
    if i == 0 and np.isnan(right_hand_landmarks).all():
        right_hand_landmarks = np.zeros((21, 3))
    face_landmarks = extract_landmarks(row, "face", 468)

    # Scale all parts using a unified bounding box
    pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks = scale_all_landmarks(
        pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
    )

    # Smooth movement if previous frame data exists
    if prev_frame:
        pose_landmarks = interpolate_landmarks(prev_frame["pose"], pose_landmarks, 0.4)
        left_hand_landmarks = interpolate_landmarks(prev_frame["left_hand"], left_hand_landmarks, 0.4)
        right_hand_landmarks = interpolate_landmarks(prev_frame["right_hand"], right_hand_landmarks, 0.4)
        face_landmarks = interpolate_landmarks(prev_frame["face"], face_landmarks, 0.4)

    draw_skeleton(img, pose_landmarks, POSE_CONNECTIONS, (0, 255, 0))
    draw_skeleton(img, left_hand_landmarks, HAND_CONNECTIONS, (255, 0, 0))
    draw_skeleton(img, right_hand_landmarks, HAND_CONNECTIONS, (0, 0, 255))
    draw_landmarks(img, pose_landmarks, (0, 255, 0))
    # draw_landmarks(img, left_hand_landmarks, (255, 0, 0))
    # draw_landmarks(img, right_hand_landmarks, (0, 0, 255))
    # draw_landmarks(img, face_landmarks, (0, 165, 255))

    if prev_frame:
        # Optional cross-fade effect
        img = cv2.addWeighted(prev_frame["img"], 0.5, img, 0.5, 0)

    prev_frame = {
        "img": img,
        "pose": pose_landmarks,
        "left_hand": left_hand_landmarks,
        "right_hand": right_hand_landmarks,
        "face": face_landmarks
    }

    cv2.imshow("Pose Animation", img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()