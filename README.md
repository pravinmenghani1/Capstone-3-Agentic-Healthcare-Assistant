# 🏥 Agentic Healthcare Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive AI-powered healthcare assistant demonstrating **Agentic AI**, **RAG pipelines**, and **multi-agent systems** for automated medical task management.

## 🎯 What This Project Demonstrates

### **Agentic AI System Design**
- Multi-agent coordination using LangGraph
- Task planning and goal decomposition
- Tool integration and memory management
- Complete agent execution workflows

### **RAG Pipeline Implementation**
- FAISS vector database for medical knowledge
- Semantic search with sentence transformers
- Context-aware patient information retrieval
- Medical document processing and summarization

### **LLMOps Integration**
- Model evaluation and quality assessment
- Performance monitoring and analytics
- Interactive dashboard with real-time metrics
- Comprehensive logging and error handling

## 🚀 Quick Start

### Prerequisites
**📋 IMPORTANT:** Complete the setup from [prereqs.md](prereqs.md) before proceeding.

### 1. Clone and Setup
```bash
git clone https://github.com/pravinmenghani1/Capstone-3-Agentic-Healthcare-Assistant.git
cd Capstone-3-Agentic-Healthcare-Assistant

# Create virtual environment
python -m venv healthcare_env
source healthcare_env/bin/activate  # On Windows: healthcare_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Launch the interactive dashboard
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

## 🤖 LLM Configuration

Choose your preferred LLM provider:

### **Option A: OpenAI (Cloud)**
1. Get API key from [OpenAI Platform](https://platform.openai.com)
2. Select "OpenAI GPT-4" in the sidebar
3. Enter your API key

### **Option B: Ollama (Local)**
1. Install from [ollama.ai](https://ollama.ai)
2. Download a model: `ollama pull llama3.1`
3. Select "Ollama (Local)" in the sidebar

### **Option C: Mock LLM (Testing)**
- Select "Mock LLM (Testing)" - no setup required

## 🏗️ Architecture

```
🧠 Multi-Agent System
├── 📋 Planner Agent - Task decomposition and planning
├── 📅 Appointment Agent - Medical appointment booking
├── 📄 Records Agent - Patient data management
└── 🔍 Search Agent - Medical information retrieval

🔧 Core Components
├── 💾 Vector Memory - FAISS-based patient context
├── 🛠️ Tool Integration - Medical APIs and databases
├── 📊 Evaluation System - LLMOps monitoring
└── 🌐 Web Interface - Streamlit dashboard
```

## 📊 Sample Data

The project includes real patient data for demonstration:

- **Ramesh Kulkarni** (65, Male) - Hypertension
- **Anjali Mehra** (33, Female) - Upper Respiratory Infection  
- **David Thompson** (51, Male) - Type 2 Diabetes
- **Rahul Negi** (31, Male) - Healthy

## 🎯 Usage Examples

### **Multi-Task Query**
```
Input: "My father has kidney disease. Book a nephrologist appointment and explain treatment options."

System Response:
✅ Plans multi-step execution
✅ Retrieves patient context  
✅ Books specialist appointment
✅ Searches latest treatments
✅ Provides comprehensive response
```

### **Patient Records**
```
Input: "Show medical history for Ramesh"

Output:
📋 Medical Records Found for Ramesh Kulkarni
👤 Age: 65 years, Male
🏥 Condition: Hypertension
📝 Clinical Summary: [Detailed medical information]
```

## 📁 Project Structure

```
Capstone-3-Agentic-Healthcare-Assistant/
├── streamlit_app.py              # Main application
├── requirements.txt              # Dependencies
├── prereqs.md                   # Setup prerequisites
├── dataset/                     # Sample medical data
│   ├── records.xlsx            # Patient records
│   └── *.pdf                   # Medical reports
├── src/                        # Source code
│   ├── agents/                 # AI agents
│   ├── tools/                  # Agent tools
│   ├── memory/                 # Memory systems
│   └── evaluation/             # LLMOps monitoring
└── examples/                   # Demo scripts
    ├── simple_demo.py          # Basic functionality
    └── comprehensive_demo.py   # Full system demo
```

## 🔧 Configuration

### **Environment Variables (Optional)**
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
```

### **System Settings**
- **Temperature**: 0.1 (consistent responses)
- **Vector DB**: FAISS with 384-dimensional embeddings
- **Models**: Automatic detection for Ollama

## 🛠️ Development

### **Run Examples**
```bash
# Test basic functionality
python examples/simple_demo.py

# Run comprehensive demo
python examples/comprehensive_demo.py
```

### **Add New Agents**
1. Create agent class in `src/agents/`
2. Implement `execute()` method
3. Register in main coordinator

## 📊 Features Checklist

### **✅ Agentic AI System**
- [x] Multi-agent coordination with LangGraph
- [x] Task planning and goal decomposition
- [x] Tool integration and API management
- [x] Memory systems with context awareness

### **✅ RAG Pipeline**
- [x] Vector database with FAISS
- [x] Semantic similarity search
- [x] Medical document processing
- [x] Context-aware information retrieval

### **✅ LLMOps Integration**
- [x] Model evaluation and scoring
- [x] Performance monitoring dashboard
- [x] Error handling and logging
- [x] Interactive analytics interface

## 🔒 Safety & Compliance

- **Medical Disclaimers**: All responses include appropriate warnings
- **Data Privacy**: Local data storage with no external transmission
- **Educational Purpose**: Designed for learning, not medical diagnosis

## 🚀 Deployment

### **Local Development**
```bash
streamlit run streamlit_app.py
```

### **Cloud Deployment**
- **AWS**: ECS, Lambda, or EC2
- **Google Cloud**: Cloud Run
- **Azure**: Container Instances

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📚 Learning Outcomes

This project teaches:
- **Agentic AI**: Multi-agent system design and coordination
- **RAG Systems**: Vector databases and semantic search
- **LLMOps**: Model evaluation and monitoring
- **Healthcare AI**: Domain-specific AI applications
- **System Integration**: End-to-end AI system development

## 🐛 Troubleshooting

### **Common Issues**
- **PyTorch Errors**: `pip install --upgrade torch sentence-transformers`
- **Ollama Not Found**: Check `ollama list` and ensure service is running
- **Port Issues**: Use `streamlit run streamlit_app.py --server.port 8502`

### **Getting Help**
- Check [Issues](https://github.com/pravinmenghani1/Capstone-3-Agentic-Healthcare-Assistant/issues)
- Review [prereqs.md](prereqs.md) for setup
- Run `python examples/simple_demo.py` for testing

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **LangChain** for agent framework
- **Streamlit** for web interface
- **FAISS** for vector search
- **OpenAI** and **Ollama** for LLM integration

---

**⚠️ Medical Disclaimer**: This is an educational project for demonstrating AI capabilities. Not for actual medical use. Always consult healthcare professionals for medical advice.

**🎓 Educational Use**: Perfect for learning about agentic AI, RAG pipelines, and healthcare AI applications.
