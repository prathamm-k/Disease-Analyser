# PowerShell setup script for Disease Analyser
Write-Host "Setting up Disease Analyser Project..." -ForegroundColor Cyan

# Check if Python is installed
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python is not installed! Please install Python 3.7 or higher." -ForegroundColor Red
    exit 1
}

# Check if pip is installed
$pip = Get-Command pip -ErrorAction SilentlyContinue
if (-not $pip) {
    Write-Host "pip is not installed! Please install pip." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "disease-venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv disease-venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\disease-venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install dependencies!" -ForegroundColor Red
    exit 1
}

# Run the training script
if (Test-Path "train_model/train_model.py") {
    Write-Host "Running model training script..." -ForegroundColor Yellow
    .\disease-venv\Scripts\python.exe train_model\train_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Model training failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "train_model.py not found! Skipping model training." -ForegroundColor Red
}

Write-Host "\nSetup completed successfully!\n" -ForegroundColor Green
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment if not already activated: .\disease-venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "2. Run the training script if not already run during setup: python train_model/train_model.py" -ForegroundColor Cyan
Write-Host "3. Run the application: streamlit run app.py" -ForegroundColor Cyan 