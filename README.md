# Disease Predictor

## Project Motivation
This project was created as a learning exercise to understand Machine Learning and Data Modeling in real-world applications. The goal is to explore how ML can be used to solve practical problems in healthcare and society. Through this project, I aim to gain hands-on experience with:
- Data preprocessing and feature engineering
- Machine Learning model training and evaluation
- Building interactive web applications
- Understanding how ML can be applied to healthcare diagnostics

## Project Overview
The Disease Predictor is a web application that uses Machine Learning to predict potential diseases based on user-reported symptoms. The system:
- Uses a Random Forest Classifier trained on a comprehensive dataset of symptoms and diseases
- Allows users to input up to 5 symptoms
- Provides instant predictions about potential diseases
- Features a user-friendly interface built with Streamlit

## Features
- Interactive symptom selection
- Real-time disease prediction
- Support for 41 different diseases
- Analysis of 132 different symptoms
- High-accuracy predictions using Random Forest algorithm

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Git

## Project Structure
```
Disease-Analyser/
  ├── app.py
  ├── disease-venv/
  ├── README.md
  ├── requirements.txt
  ├── setup.sh
  ├── setup.ps1
  └── train_model/
      ├── models/
      │   ├── diseases.pkl
      │   ├── random_forest_model.pkl
      │   └── symptoms.pkl
      ├── Testing.csv
      ├── train_model.log
      ├── train_model.py
      └── Training.csv
```

## Quick Setup

### For Windows Users
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Disease-Analyser.git
cd Disease-Analyser
```
2. Run the setup script:
```bash
setup.ps1
```
3. After setup completes, start the application:
```bash
disease-venv\Scripts\activate
streamlit run app.py
```

### For Arch Linux Users
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Disease-Analyser.git
cd Disease-Analyser
```
2. Make the setup script executable and run it:
```bash
chmod +x setup.sh
./setup.sh
```
3. After setup completes, start the application:
```bash
source disease-venv/bin/activate
streamlit run app.py
```

> **Note:** The setup scripts will automatically install dependencies and run the model training script (`train_model/train_model.py`) before you start the app. If you set up manually, you must run the training script yourself before running the app.

## Manual Setup (Alternative Method)
If you prefer to set up manually or if the scripts don't work:

### For Windows Users
1. Open Command Prompt or PowerShell
2. Clone the repository:
```bash
git clone https://github.com/yourusername/Disease-Analyser.git
cd Disease-Analyser
```
3. Create and activate virtual environment:
```bash
python -m venv disease-venv
disease-venv\Scripts\activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. **Run the model training script:**
```bash
python train_model\train_model.py
```
6. Run the application:
```bash
streamlit run app.py
```

### For Arch Linux Users
1. Open Terminal
2. Clone the repository:
```bash
git clone https://github.com/yourusername/Disease-Analyser.git
cd Disease-Analyser
```
3. Create and activate virtual environment:
```bash
python -m venv disease-venv
source disease-venv/bin/activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. **Run the model training script:**
```bash
python train_model/train_model.py
```
6. Run the application:
```bash
streamlit run app.py
```

## Usage
1. After starting the application, a web interface will open in your default browser
2. Select up to 5 symptoms from the dropdown menus
3. Click the "Predict Disease" button
4. The system will display the predicted disease based on your symptoms

## Technical Details
- Built with Python and Streamlit
- Uses scikit-learn for machine learning
- Implements Random Forest Classifier
- Data preprocessing includes symptom encoding and label mapping

## Dataset
The project uses two datasets:
- Training.csv: Contains the training data for the model
- Testing.csv: Contains the test data for model evaluation

## Dependencies
- streamlit
- pandas
- numpy
- scikit-learn

## Future Improvements
- Add more diseases and symptoms
- Implement multiple ML models for comparison
- Add confidence scores for predictions
- Include detailed disease information and recommendations
- Add user authentication and history tracking

## Contributing
Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License
This project is open source and available under the MIT License. 