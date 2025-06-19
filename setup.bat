@echo off
echo Setting up Disease Predictor Project...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.7 or higher.
    exit /b 1
)

:: Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed! Please install pip.
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "disease-venv" (
    echo Creating virtual environment...
    python -m venv disease-venv
)

:: Activate virtual environment
echo Activating virtual environment...
call disease-venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Check if installation was successful
if errorlevel 1 (
    echo Failed to install dependencies!
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To start the application:
echo 1. Activate the virtual environment: disease-venv\Scripts\activate
echo 2. Run the application: streamlit run app.py
echo.
pause 