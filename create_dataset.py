import os

dataset_folder = "C:\\Users\\Roy\\Documents\\small_dataset\\raw_videos"
processed_folder = "C:\\Users\\Roy\\Documents\\small_dataset\\landmarks"

for video in os.listdir(dataset_folder):
    if video.endswith(".mp4"):
        video_path = os.path.join(dataset_folder, video)
        print("Processing video:", video_path)
        # Call the function to process the video
        os.system("python mediapipe_holistic.py --video " + video_path + " --output " + processed_folder + "\\ --save_together")
