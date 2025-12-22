# Prerequisites for Agentic Healthcare Assistant
📧 Send this to learners BEFORE the session to ensure they're ready!

## 🎯 What You Need to Prepare

### 1. Python 3.8+ Installation

**Windows:**
- Go to [python.org/downloads](https://python.org/downloads)
- Download Python 3.8+ (latest recommended)
- **IMPORTANT:** Check "Add Python to PATH" during installation
- Test: Open Command Prompt and type `python --version`

**macOS:**
```bash
# Option 1: Download from python.org (recommended for beginners)
# Go to python.org/downloads and download the installer

# Option 2: Using Homebrew (if you have it)
brew install python
```
Test: Open Terminal and type `python3 --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install python3 python3-pip
# or for newer versions: sudo dnf install python3 python3-pip
```
Test: Type `python3 --version`

### 2. Git Installation

**Windows:**
- Download from [git-scm.com](https://git-scm.com)
- Install with default settings

**macOS:**
```bash
# Option 1: Download from git-scm.com
# Option 2: Install Xcode Command Line Tools
xcode-select --install
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install git

# CentOS/RHEL
sudo yum install git
```

### 3. LLM Options (Choose One)

#### Option A: OpenAI API (Cloud-based, Paid)
**CRITICAL - Do this before the session:**

1. **Create Account:** Go to [platform.openai.com](https://platform.openai.com)
   - Sign up or log in
2. **Add Payment Method:** Go to Billing → Add payment method
3. **Generate API Key:**
   - Go to API Keys section
   - Click "Create new secret key"
   - **COPY AND SAVE IT SECURELY** (you won't see it again!)
   - Format looks like: `sk-proj-...` (long string)

💰 **Cost:** Minimal usage (~$1-3 for the entire project)

#### Option B: Ollama (Local, Free)
**Install Ollama for local LLM:**

1. **Download:** Go to [ollama.ai](https://ollama.ai)
2. **Install** the application for your OS
3. **Pull a model:**
```bash
# After installation, run:
ollama pull llama3.1
# or
ollama pull gemma2
```
4. **Test:** `ollama list` should show your downloaded models

💡 **Recommendation:** Ollama is free but requires more system resources. OpenAI is easier for beginners.

### 4. Code Editor (Optional but Recommended)
Choose one:
- **VS Code:** [code.visualstudio.com](https://code.visualstudio.com) (recommended)
- **PyCharm Community:** [jetbrains.com/pycharm](https://jetbrains.com/pycharm)
- Any text editor you prefer

## ✅ Pre-Session Checklist
Complete these BEFORE the session:

- [ ] Python 3.8+ installed and working (`python --version` or `python3 --version`)
- [ ] Git installed and working (`git --version`)
- [ ] **Either:** OpenAI account + API key + payment method **OR** Ollama installed with a model
- [ ] Code editor installed (optional)
- [ ] Stable internet connection

## 🧪 Test Your Setup
Run these commands to verify everything works:

```bash
# Test Python
python --version
# or on Mac/Linux:
python3 --version

# Test pip
pip --version
# or:
pip3 --version

# Test Git
git --version

# Test virtual environment creation
python -m venv test_env
# Clean up
rmdir /s test_env  # Windows
rm -rf test_env    # Mac/Linux

# If using Ollama, test:
ollama list
```

## 🚨 Common Issues & Quick Fixes

**"Python not found":**
- Windows: Reinstall Python with "Add to PATH" checked
- Mac/Linux: Use `python3` instead of `python`

**"Permission denied":**
- Windows: Run Command Prompt as Administrator
- Mac/Linux: Use `sudo` for system installations

**OpenAI API Key Issues:**
- Make sure you've added a payment method
- API key should start with `sk-proj-` or `sk-`
- Keep it secure - never share it!

**Ollama Issues:**
- Make sure Ollama service is running
- Try `ollama serve` if models aren't responding
- Ensure you have at least 8GB RAM for larger models

## 📱 What to Bring to the Session
- Your **OpenAI API Key** (if using OpenAI) saved securely
- **OR** Ollama installed with a downloaded model
- Laptop with all software installed
- Stable internet connection
- Enthusiasm to learn! 🚀

## ❓ Need Help?
If you encounter issues during setup:
- Try the common fixes above
- Search for the specific error message online
- Ask for help in our discussion forum
- Come to the session - we'll help you get set up!

## 🎯 Session Day
On the day of the session, we'll:
1. ✅ Verify your setup quickly
2. 📥 Clone the project repository
3. 🐍 Set up the virtual environment
4. 🔑 Configure your LLM (OpenAI API key or Ollama)
5. 🚀 Run the Agentic Healthcare Assistant!

## 🏥 What You'll Build
You're going to create an amazing **multi-agent AI healthcare system** that can:
- 📅 Book medical appointments automatically
- 📋 Manage patient medical records with AI summarization
- 🔍 Search medical information using RAG (Retrieval-Augmented Generation)
- 🧠 Coordinate multiple AI agents using LangGraph
- 📊 Monitor system performance with LLMOps

**Technologies you'll learn:**
- Agentic AI system design
- RAG pipelines with vector databases
- Multi-agent coordination
- LLMOps and model evaluation
- Healthcare AI applications

You're going to build something incredible! 🤖✨

---

**⚠️ Medical Disclaimer:** This is an educational project for learning AI concepts. The system should not be used for actual medical decision-making. Always consult qualified healthcare professionals for medical advice.
