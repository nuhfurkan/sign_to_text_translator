# Export Landmards

This repository includes various landmark detection models to export the landmarks from the bible videos. The files with <<method_name>>.py are the processing files. The file main.py has a number of terminal tools to see and test on the data. 

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
or use setup file: 

```
./setup.sh
```

## CLI Reference

main.py implements a simple cli. 
Use the commands always in following format
```
python main.py [option] [flags]
```

The available commands and options are given as follows:
- `frame` diplays
- 

