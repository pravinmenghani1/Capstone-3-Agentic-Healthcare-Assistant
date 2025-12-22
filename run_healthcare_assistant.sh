#!/bin/bash

# Healthcare Assistant Setup Script
echo "🏥 Agentic Healthcare Assistant - Setup & Run"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "healthcare_env" ]; then
    echo "Creating virtual environment..."
    python -m venv healthcare_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source healthcare_env/bin/activate

# Install requirements if needed
if [ ! -f "healthcare_env/pyvenv.cfg" ] || [ ! -f "healthcare_env/lib/python*/site-packages/streamlit" ]; then
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo ""
echo "✅ Setup complete! Choose an option:"
echo ""
echo "1. Run Simple Demo (python simple_demo.py)"
echo "2. Run Streamlit Web App (streamlit run streamlit_app.py)"
echo "3. Run Full System Demo (python demo.py)"
echo "4. Activate environment only"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Running simple demo..."
        python simple_demo.py
        ;;
    2)
        echo "Starting Streamlit web application..."
        echo "Open http://localhost:8501 in your browser"
        streamlit run streamlit_app.py
        ;;
    3)
        echo "Running full system demo..."
        python demo.py
        ;;
    4)
        echo "Virtual environment activated. Run commands manually."
        echo "To deactivate, type: deactivate"
        exec bash
        ;;
    *)
        echo "Invalid choice. Virtual environment is activated."
        echo "Run: python simple_demo.py OR streamlit run streamlit_app.py"
        ;;
esac
