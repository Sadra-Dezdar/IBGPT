# IB Student Assistant 🎓

An intelligent AI-powered assistant designed specifically for International Baccalaureate (IB) students. This system provides comprehensive support for IB programme queries, Internal Assessment (IA) feedback, and academic guidance using advanced multi-agent AI architecture.

## ✨ Features

### Core Capabilities
- **Intelligent Query Processing**: Multi-agent system for sophisticated question handling
- **IA Assessment**: Automated analysis and feedback for Internal Assessments
- **Document Analysis**: Support for PDF and DOCX file uploads
- **Subject-Specific Guidance**: Tailored advice for all IB subjects
- **Real-time Chat Interface**: Interactive web-based interface

### AI Architecture
- **Fast Agent (Qwen3)**: Quick query classification and routing
- **RAG Agent (DeepSeek-R1)**: Deep reasoning with document retrieval
- **Consensus Agent (DeepSeek-R1)**: Response review and refinement

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama server running locally
- Required AI models (automatically downloaded)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ib-student-assistant.git
   cd ib-student-assistant
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama server**
   ```bash
   ollama serve
   ```

5. **Download required models**
   ```bash
   ollama pull qwen3:latest
   ollama pull deepseek-r1:14b-qwen-distill-q4_K_M
   ```

### Running the Application

**Option 1: Simple startup (Recommended)**
```bash
python run_app.py
```

**Option 2: Status checker with cleanup**
```bash
python start_app.py
```

# IB Student Assistant 🎓

An intelligent AI-powered assistant designed specifically for International Baccalaureate (IB) students. This system provides comprehensive support for IB programme queries, Internal Assessment (IA) feedback, and academic guidance using advanced multi-agent AI architecture.

## ✨ Features

### Core Capabilities
- **Intelligent Query Processing**: Multi-agent system for sophisticated question handling
- **IA Assessment**: Automated analysis and feedback for Internal Assessments
- **Document Analysis**: Support for PDF and DOCX file uploads
- **Subject-Specific Guidance**: Tailored advice for all IB subjects
- **Real-time Chat Interface**: Interactive web-based interface

### AI Architecture
- **Fast Agent (Qwen3)**: Quick query classification and routing
- **RAG Agent (DeepSeek-R1)**: Deep reasoning with document retrieval
- **Consensus Agent (DeepSeek-R1)**: Response review and refinement

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama server running locally
- Required AI models (automatically downloaded)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ib-student-assistant.git
   cd ib-student-assistant
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama server**
   ```bash
   ollama serve
   ```

5. **Download required models**
   ```bash
   ollama pull qwen3:latest
   ollama pull deepseek-r1:14b-qwen-distill-q4_K_M
   ```

### Running the Application

**Option 1: Simple startup (Recommended)**
```bash
python run_app.py
```

**Option 2: Status checker with cleanup**
```bash
python start_app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
ib-student-assistant/
├── agents/                      # AI agent implementations
│   └── multi_agent_system_no_tools.py
├── config/                      # Configuration files
│   ├── collections.py           # ChromaDB collection setup
│   ├── metadata.py              # Document metadata definitions
│   └── settings.py              # Application settings
├── core/                        # Core processing modules
│   ├── chunker.py               # Document chunking
│   ├── document_processor.py    # Document processing
│   └── retrieval.py             # Information retrieval
├── data/                        # Document collections
│   ├── general_info/            # General IB information
│   ├── ia_guides/               # IA guides by subject
│   ├── mark_schemes/            # Exam mark schemes
│   ├── past_papers/             # Past examination papers
│   └── syllabi/                 # Subject syllabi
├── interface/                   # Web interface
│   └── streamlit_app.py         # Main Streamlit application
├── scripts/                     # Utility scripts
│   ├── batch_ingest.py          # Batch document ingestion
│   ├── ingest_documents.py      # Document ingestion
│   └── setup_collections.py    # Database setup
├── utils/                       # Utility modules
│   ├── chat_manager.py          # Chat session management
│   ├── chromadb_utils.py        # Database utilities
│   ├── ia_assessor.py           # IA assessment logic
│   ├── pdf_upload_handler.py    # File upload handling
│   └── text_processing.py       # Text processing utilities
├── chroma_db/                   # Document database (auto-created)
├── temp_uploads/                # Temporary file storage
├── main.py                      # CLI interface
├── manage.py                    # Management commands
├── run_app.py                   # Application launcher
├── start_app.py                 # Alternative launcher
└── requirements.txt             # Python dependencies
```

## 💬 Usage Guide

### Basic Queries
Ask questions about the IB programme:
- "What are the IB core requirements?"
- "How is the Extended Essay graded?"
- "What's the difference between HL and SL?"

### IA Assessment
Upload your IA documents for analysis:
1. Click "Upload Document" in the sidebar
2. Select your PDF or DOCX file
3. Provide a brief description
4. Get detailed feedback and suggestions

### Subject-Specific Help
Get targeted guidance:
- "Mathematics IA ideas for HL"
- "Physics Extended Essay topics"
- "History IA research methods"

## 🔧 Configuration

### Environment Variables
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)

### Model Configuration
The system uses optimized models for different tasks:
- **Fast Agent**: `qwen3:latest` - Quick classification
- **RAG Agent**: `deepseek-r1:14b-qwen-distill-q4_K_M` - Deep reasoning
- **Consensus Agent**: `deepseek-r1:14b-qwen-distill-q4_K_M` - Response refinement

### Document Collections
The system organizes documents into specialized collections:
- `ib_general`: Core IB programme information
- `ia_guides`: Subject-specific IA guidance
- `ia_examples`: Exemplar IAs with feedback
- `mark_schemes`: Exam marking criteria
- `syllabus`: Official curriculum guides

## 🛠 Development

### Adding New Documents
```bash
python scripts/ingest_documents.py --help
```

### Managing the Database
```bash
python manage.py --help
```

### Available Management Commands
- `clear-history`: Clear chat history
- `rebuild-db`: Rebuild document database
- `list-collections`: Show all collections
- `status`: Check system health
- `test`: Run comprehensive tests

## 📊 Performance

- **Response Time**: < 10 seconds for complex queries
- **Document Retrieval**: Sub-second search across 1000+ documents
- **Concurrent Users**: Supports multiple simultaneous sessions
- **Accuracy**: High-quality responses validated against IB standards

## 🔧 Troubleshooting

### App Won't Start
```bash
# Kill any existing processes
pkill -f streamlit

# Use the status checker
python start_app.py
```

### Models Not Loading
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list
```

### Dependencies Missing
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Contact the development team

---

**Made with ❤️ for IB students worldwide**