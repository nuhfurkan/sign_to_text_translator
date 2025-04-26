# Export Landmards



## Setup

Create a virtual env with following code
```
python -m venv export_landmarks_env
```

Activate the virtual environmnet on linux

```
./export_landmarks_env/bin/activate
```

or on windows

```
.\export_landmarks_env\Scripts\activate
```

Download necessary libraries using following command
```
pip install -r requirements.txt
```

## Usage

### mediapipe_holistic
Call mediapipe_holistic.py with following command.
This script processes a video file to extract landmarks and outputs the data into a timestamped CSV file.

---
```bash
python mediapipe_holistic.py --video ./data/videos/greet.mp4 --output ./data/greet_output/ --save_separate
```

### pre_processing
This script preprocesses landmark data extracted from video input, handling missing values, normalization, transformations, and more.

---

#### 🚀 Overview

The preprocessing pipeline performs the following steps sequentially:

1. **Read** the input landmark data CSV file.
2. **Skip** initial blank frames (if any).
3. **Impute** missing entries (twice for robustness).
4. **Detect** and handle outliers.
5. **Translate** hand landmarks for positional alignment.
6. **Smooth** the data to reduce jitter.
7. **Rotate** the landmarks for consistent orientation.
8. **Normalize** the data for further analysis or model input.

Each transformation can optionally be saved to a file using corresponding command-line flags.

---

#### 📦 Usage

```bash
python preprocess.py --data ./data/landmarks_output.csv 
```
### motifs

This script finds **motifs** (repeated patterns) in time series landmark data, such as motion or gesture sequences. Optionally, it can plot the motifs and save them to a CSV file.

---

#### 🚀 Overview

The motif discovery pipeline performs the following:

1. Loads time series data (default: preprocessed landmark data).
2. Searches for repeated subsequences (motifs) using a specified window size `--m`.
3. Optionally plots the motifs and their nearest neighbors.
4. Optionally saves the motifs and metadata to a CSV file.

---

#### 📦 Usage

```bash
python motifs.py --m 40 --data ./data/normalized_landmarks.csv --plot --save ./output/motifs.csv
```

## Visualization
### visualise.py
Visualise the landmarks on given video input.

### reconstruct.py
Reconstruct a stickman out of generated landmarks.

### visualise_motifs.py
Visualise selected columns for processed motifs data.

## PCA
```bash
python pca.py --data ./data/path_to_file.csv --save ./data/principals
```