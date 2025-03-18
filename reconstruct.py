import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("landmarks_output.csv")

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Right arm
    (0, 4), (4, 5), (5, 6), (6, 8),  # Left arm
    (9, 10), (11, 12),              # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left leg
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22)   # Right leg
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
    return np.array(landmarks)

def scale_all_landmarks(pose, left_hand, right_hand, face, width=1280, height=720):
    all_points = []
    valid_arrays = []
    for arr in [pose, left_hand, right_hand, face]:
        if arr is not None and arr.size > 0:
            all_points.append(arr)
            valid_arrays.append(arr.shape[0])
    if not all_points:
        return pose, left_hand, right_hand, face

    combined = np.concatenate(all_points, axis=0)

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

    split_arrays = []
    idx = 0
    for count in valid_arrays:
        slice_arr = combined[idx: idx + count]
        idx += count
        split_arrays.append(slice_arr)

    new_pose, new_left, new_right, new_face = None, None, None, None
    arr_idx = 0
    if pose is not None and pose.size > 0:
        new_pose = split_arrays[arr_idx]
        arr_idx += 1
    if left_hand is not None and left_hand.size > 0:
        new_left = split_arrays[arr_idx]
        arr_idx += 1
    if right_hand is not None and right_hand.size > 0:
        new_right = split_arrays[arr_idx]
        arr_idx += 1
    if face is not None and face.size > 0:
        new_face = split_arrays[arr_idx]

    return new_pose, new_left, new_right, new_face

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
    return prev * (1 - alpha) + new * alpha

image_size = (720, 1280, 3)
prev_frame = None

for i in range(len(df)):
    img = np.ones(image_size, dtype=np.uint8) * 255
    row = df.iloc[i]

    pose_landmarks = extract_landmarks(row, "pose", 33)
    left_hand_landmarks = extract_landmarks(row, "left_hand", 21)
    right_hand_landmarks = extract_landmarks(row, "right_hand", 21)
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
    draw_landmarks(img, left_hand_landmarks, (255, 0, 0))
    draw_landmarks(img, right_hand_landmarks, (0, 0, 255))
    draw_landmarks(img, face_landmarks, (0, 165, 255))

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