import os
import cv2
import time
import mediapipe as mp
import pandas as pd
import csv
import argparse

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def get_video_info(video_path: str):
    """
    Get video information such as width, height, and FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return width, height, fps


def process_video(video_path :str, csv_folder: str,only_hands: bool = False, only_pose: bool = False, save_together: bool = False):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(csv_folder):
        os.mkdir(csv_folder)
    
    # Landmark names for CSV headers
    pose_landmarks = [f'pose_{i}_{coord}' for i in range(33) for coord in ['x', 'y', 'z']]
    left_hand_landmarks = [f'left_hand_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    right_hand_landmarks = [f'right_hand_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    # face_landmarks = [f'face_{i}_{coord}' for i in range(468) for coord in ['x', 'y', 'z']]

    columns = []
    # Column headers
    if only_hands:
        columns = ["frame"] + left_hand_landmarks + right_hand_landmarks
    elif only_pose:
        columns = ["frame"] + pose_landmarks
    else:
        columns = ["frame"] + pose_landmarks + left_hand_landmarks + right_hand_landmarks  # + face_landmarks

    # Create a CSV file and write the header
    if not save_together:
        with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "_pose.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pose_landmarks)
        with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "_left_hand.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(left_hand_landmarks)
        with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "_right_hand.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(right_hand_landmarks)
    else:
        with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "_landmarks.csv", mode='w', newline='') as file:
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
            pose_landmarks = [frame_count]
            right_hand_landmarks = [frame_count]
            left_hand_landmarks = [frame_count]

            # Pose landmarks
            if not only_hands:
                if results.pose_landmarks:
                    pose_landmarks += [val for lm in results.pose_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
                else:
                    pose_landmarks += [None] * (33 * 3)

            # Left hand landmarks
            if results.left_hand_landmarks:
                left_hand_landmarks += [val for lm in results.left_hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
            else:
                left_hand_landmarks += [None] * (21 * 3)

            # Right hand landmarks
            if results.right_hand_landmarks:
                right_hand_landmarks += [val for lm in results.right_hand_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
            else:
                right_hand_landmarks += [None] * (21 * 3)

            # Face landmarks
            # if results.face_landmarks:
            #     frame_landmarks += [val for lm in results.face_landmarks.landmark for val in [lm.x, lm.y, lm.z]]
            # else:
            #     frame_landmarks += [None] * len(face_landmarks)

            # print(frame_landmarks)

            # Save to CSV
            if not save_together:
                if not only_hands:
                    with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "pose.csv", mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(pose_landmarks)
                with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "left_hand.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(left_hand_landmarks)
                with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "right_hand.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(right_hand_landmarks)
            else:
                with open(csv_folder + "\\" + os.path.basename(video_path)[:-4] + "_landmarks.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(pose_landmarks + left_hand_landmarks[1:] + right_hand_landmarks[1:])

            frame_count += 1

            # print(f"Frame {frame_count}")

    elapsed_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")

    # Release resources
    cap.release()
    print(f"Landmarks saved to {csv_folder}")

if __name__ == "__main__":
    # Output CSV file
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    csv_folder = "./data/landmarks_output_" + date

    parser = argparse.ArgumentParser(description="Process video and extract landmarks.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--output", type=str, default=csv_folder, help="Path to the output CSV folder.")
    parser.add_argument("--only_hands", default=False, action="store_true", help="Process only hands.")
    parser.add_argument("--only_pose", default=False, action="store_true", help="Process only pose.")
    parser.add_argument("--save_together", default=False, action="store_true", help="Save together CSV files for each type of landmark.")

    parser.add_argument("--video_info", default=False, action="store_true", help="Get video information (width, height, fps).")

    args = parser.parse_args()

    if args.video_info:
        video_path = args.video
        print(f"Getting video info for: {video_path}")
        video_info = get_video_info(video_path)
        if video_info:
            width, height, fps = video_info
            print(f"Width: {width}, Height: {height}, FPS: {fps}")
        else:
            print("Could not retrieve video information.")

    elif args.video:
        video_path = args.video
        output_path = args.output
        print(f"Processing video: {video_path}")

        if args.only_hands and args.only_pose:
            print("Cannot select both hands and pose. Please choose one.")
            exit(1)            

        process_video(video_path, output_path, only_hands=args.only_hands, only_pose=args.only_pose, save_together=args.save_together)
    else:
        print("No video file provided. Using default video.")

# video_path = "./data/videos/greet.mp4"  # Change to your video file path