#!/bin/bash

echo "Setting up the environment for the project"

echo "Creating the virtual environment"
python -m venv export_landmarks_env

echo "Activating the virtual environment"
./export_landmarks_env/bin/activate

echo "Installing the required packages"
pip install -r requirements.txt

echo "Environment setup is complete"