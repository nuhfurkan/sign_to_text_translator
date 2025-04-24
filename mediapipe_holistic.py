import cv2
import time
import mediapipe as mp
import pandas as pd
import csv
import argparse

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic


def process_video(video_path :str, csv_file: str):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Landmark names for CSV headers
    pose_landmarks = [f'pose_{i}_{coord}' for i in range(33) for coord in ['x', 'y', 'z']]
    left_hand_landmarks = [f'left_hand_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    right_hand_landmarks = [f'right_hand_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    # face_landmarks = [f'face_{i}_{coord}' for i in range(468) for coord in ['x', 'y', 'z']]

    # Column headers
    columns = ["frame"] + pose_landmarks + left_hand_landmarks + right_hand_landmarks  # + face_landmarks

    # Create a CSV file and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)

    # Process video
    frame_count = 0
    with mp_holistic.Holistic(
        static_image_mode=False, 
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        model_complexity=1,
        ) as holistic:

        print("Processing video...")
        start_time = time.time()
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
                frame_landmarks += [val for lm in results.pose_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
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
            # if results.face_landmarks:
            #     frame_landmarks += [val for lm in results.face_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
            # else:
            #     frame_landmarks += [None] * len(face_landmarks)

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

if __name__ == "__main__":
    # Output CSV file
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = "./data/landmarks_output_" + date + ".csv"

    parser = argparse.ArgumentParser(description="Process video and extract landmarks.")
    parser.add_argument("--video", type=str, help="Path to the video file.")
    parser.add_argument("--output", type=str, default=csv_file, help="Path to the output CSV file.")
    args = parser.parse_args()

    if args.video:
        video_path = args.video
        output_path = args.output
        print(f"Processing video: {video_path}")

        process_video(video_path, output_path)
    else:
        print("No video file provided. Using default video.")

# video_path = "./data/videos/greet.mp4"  # Change to your video file path