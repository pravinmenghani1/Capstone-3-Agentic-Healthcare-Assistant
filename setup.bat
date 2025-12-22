@echo off
REM Agentic Healthcare Assistant - Quick Setup Script for Windows
REM This script automates the setup process for the healthcare assistant

echo 🏥 Agentic Healthcare Assistant - Quick Setup
echo ==============================================

REM Check if Python 3 is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% found

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist healthcare_env (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q healthcare_env
)

python -m venv healthcare_env

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call healthcare_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing Python packages...
pip install -r requirements.txt

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 Next steps:
echo 1. Activate the virtual environment:
echo    healthcare_env\Scripts\activate.bat
echo.
echo 2. Choose your LLM provider:
echo    • For OpenAI: Get API key from https://platform.openai.com
echo    • For Ollama: Install from https://ollama.ai and run 'ollama pull llama3.1'
echo    • For testing: Use Mock LLM (no setup required)
echo.
echo 3. Run the application:
echo    streamlit run streamlit_app.py
echo.
echo 4. Open your browser to: http://localhost:8501
echo.
echo 📚 For detailed setup instructions, see:
echo    • README.md - Complete documentation
echo    • prereqs.md - Prerequisites and troubleshooting
echo.
echo 🎉 Happy healthcare AI exploration!
echo.
pause
