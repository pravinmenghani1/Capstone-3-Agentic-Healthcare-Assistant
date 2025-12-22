@echo off
echo 🏥 Agentic Healthcare Assistant - Setup & Run
echo ==============================================

REM Check if virtual environment exists
if not exist "healthcare_env" (
    echo Creating virtual environment...
    python -m venv healthcare_env
)

REM Activate virtual environment
echo Activating virtual environment...
call healthcare_env\Scripts\activate.bat

REM Install requirements if needed
echo Installing/updating requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Setup complete! Choose an option:
echo.
echo 1. Run Simple Demo
echo 2. Run Streamlit Web App
echo 3. Run Full System Demo
echo 4. Activate environment only
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo Running simple demo...
    python simple_demo.py
) else if "%choice%"=="2" (
    echo Starting Streamlit web application...
    echo Open http://localhost:8501 in your browser
    streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    echo Running full system demo...
    python demo.py
) else if "%choice%"=="4" (
    echo Virtual environment activated. Run commands manually.
    echo To deactivate, type: deactivate
    cmd /k
) else (
    echo Invalid choice. Virtual environment is activated.
    echo Run: python simple_demo.py OR streamlit run streamlit_app.py
)

pause
