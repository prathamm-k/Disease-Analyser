#!/bin/bash

echo "Setting up Disease Analyser Project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed! Please install Python 3.7 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip is not installed! Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "disease-venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv disease-venv
fi

# Activate virtual environment
# shellcheck disable=SC1091
source disease-venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies!"
    exit 1
fi

# Run the training script
if [ -f "train_model/train_model.py" ]; then
    echo "Running model training script..."
    disease-venv/bin/python train_model/train_model.py
    if [ $? -ne 0 ]; then
        echo "Model training failed!"
        exit 1
    fi
else
    echo "train_model.py not found! Skipping model training."
fi

echo
cat << EOF
Setup completed successfully!

To start the application:
1. Activate the virtual environment: source disease-venv/bin/activate
2. Run the application: streamlit run app.py
EOF 