import cv2
import time
import mediapipe as mp
import pandas as pd
import csv

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic


# Open video file
video_path = "mathew1.1.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)

date = time.strftime("%Y-%m-%d_%H-%M-%S")

# Output CSV file
csv_file = "landmarks_output_" + date + ".csv"

# Landmark names for CSV headers
pose_landmarks = [f'pose_{i}_{coord}' for i in range(33) for coord in ['x', 'y', 'z', 'visibility']]
left_hand_landmarks = [f'left_hand_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
right_hand_landmarks = [f'right_hand_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
face_landmarks = [f'face_{i}_{coord}' for i in range(468) for coord in ['x', 'y', 'z']]

# Column headers
columns = ["frame"] + pose_landmarks + left_hand_landmarks + right_hand_landmarks # + face_landmarks

start_time = time.time()

# Create a CSV file and write the header
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)

# Process video
frame_count = 0
with mp_holistic.Holistic(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0,
    ) as holistic:

    print("Processing video...")
    while cap.isOpened():
        (f"Processing frame {frame_count}")
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = holistic.process(rgb_frame)

        # Extract landmarks
        frame_landmarks = [frame_count]

        # Pose landmarks
        if results.pose_landmarks:
            frame_landmarks += [val for lm in results.pose_landmarks.landmark for val in [lm.x, lm.y, lm.z, lm.visibility]]
        else:
            frame_landmarks += [None] * len(pose_landmarks)

        # Left hand landmarks
        if results.left_hand_landmarks:
            frame_landmarks += [val for lm in results.left_hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
        else:
            frame_landmarks += [None] * len(left_hand_landmarks)

        # Right hand landmarks
        if results.right_hand_landmarks:
            frame_landmarks += [val for lm in results.right_hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
        else:
            frame_landmarks += [None] * len(right_hand_landmarks)

        # Face landmarks
        if results.face_landmarks:
            frame_landmarks += [val for lm in results.face_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
        else:
            frame_landmarks += [None] * len(face_landmarks)

        # print(frame_landmarks)

        # Save to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(frame_landmarks)

        frame_count += 1

        # print(f"Frame {frame_count}")

elapsed_time = time.time() - start_time
print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")

# Release resources
cap.release()
print(f"Landmarks saved to {csv_file}")
