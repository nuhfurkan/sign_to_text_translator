import os
import shlex
from concurrent.futures import ProcessPoolExecutor
import argparse

# For Windows uncomment the following lines
# dataset_folder = "C:\\Users\\Roy\\Documents\\small_dataset\\raw_videos"
# processed_folder = "C:\\Users\\Roy\\Documents\\small_dataset\\landmarks"

# For Linux uncomment the following lines
dataset_folder = "/home/roy/Documents/small_dataset/raw_videos"
processed_folder = "/home/roy/Documents/small_dataset/landmarks"

# Function to process a single video
def process_video(video_filename):
    video_path = os.path.join(dataset_folder, video_filename)
    if not video_filename.endswith(".mp4"):
        return
    print("Processing video:", video_path)
    
    command = (
        f"python mediapipe_holistic.py --video {shlex.quote(video_path)} "
        f"--output {shlex.quote(processed_folder)}/ --save_together"
    )
    os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in parallel.")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=dataset_folder,
        help="Path to the folder containing the dataset videos.",
    )
    parser.add_argument(
        "--processed_folder",
        type=str,
        default=processed_folder,
        help="Path to the folder where processed videos will be saved.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processes to use.",
    )
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    processed_folder = args.processed_folder
    # Create the processed folder if it doesn't exist
    os.makedirs(processed_folder, exist_ok=True)

    video_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mp4")]

    # Choose the number of parallel processes (n). Set to os.cpu_count() or any custom number.
    n = os.cpu_count()  # or set n = 4, for example

    with ProcessPoolExecutor(max_workers=n) as executor:
        executor.map(process_video, video_files)
