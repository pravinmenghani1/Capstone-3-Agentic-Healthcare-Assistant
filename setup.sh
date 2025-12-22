#!/bin/bash

# Agentic Healthcare Assistant - Quick Setup Script
# This script automates the setup process for the healthcare assistant

set -e  # Exit on any error

echo "🏥 Agentic Healthcare Assistant - Quick Setup"
echo "=============================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION found, but Python $REQUIRED_VERSION+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "healthcare_env" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf healthcare_env
fi

python3 -m venv healthcare_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source healthcare_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate the virtual environment:"
echo "   source healthcare_env/bin/activate"
echo ""
echo "2. Choose your LLM provider:"
echo "   • For OpenAI: Get API key from https://platform.openai.com"
echo "   • For Ollama: Install from https://ollama.ai and run 'ollama pull llama3.1'"
echo "   • For testing: Use Mock LLM (no setup required)"
echo ""
echo "3. Run the application:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "4. Open your browser to: http://localhost:8501"
echo ""
echo "📚 For detailed setup instructions, see:"
echo "   • README.md - Complete documentation"
echo "   • prereqs.md - Prerequisites and troubleshooting"
echo ""
echo "🎉 Happy healthcare AI exploration!"
