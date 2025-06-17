# IBGPT: IB Student Assistant - Multi-Agent Educational System

---

## 🎯 Project Overview

### What is IBGPT?
An AI-powered educational assistant specifically designed for International Baccalaureate (IB) students, leveraging **multi-agent architecture** and **agentic RAG** to provide intelligent tutoring, IA assessment, and comprehensive academic support.

### Key Goals
- ✅ IB-specific tutoring system
- ✅ Implement multi-agent coordination with agentic RAG
- ✅ Provide real-time IB curriculum assistance
- ✅ Enable IA (Internal Assessment) feedback and evaluation
- ✅ Build an educational query system for IB students worldwide

### Reasons to Why IB Students Need This
- Complex curriculum spanning 6 subject groups
- Rigorous Internal Assessment requirements
- Need for Free solution since tutoring and grading IAs cost 100s of dollars people can't afford to pay

---

## 🏗️ Core Components & Architecture

### Essential Components

#### 🗄️ ChromaDB Vector Database
- **Purpose**: Semantic storage and retrieval of IB curriculum documents
- **Features**: 
  - Efficient similarity search across IB materials
  - Multiple collections for different document types
  - Persistent storage with metadata filtering
  - Optimized for educational content retrieval

#### 🤖 Pydantic AI Framework
- **Purpose**: Structured AI agent development and coordination
- **Benefits**:
  - Type-safe agent interactions
  - Robust error handling and validation
  - Seamless integration with language models
  - Clean agent-to-agent communication protocols

#### 🧠 Language Models Integration
- **DeepSeek**: Advanced reasoning and complex problem-solving
- **Qwen**: Fast response generation and general queries
- **Sentence Transformers**: Document embedding and similarity matching

#### 🌐 Streamlit Web Interface
- **Features**: 
  - Real-time chat interface
  - PDF upload for IA assessment
  - Document management system
  - Multi-session conversation handling

---

## 🤝 Multi-Agent System Architecture

### From Vanilla RAG to Agentic RAG

#### Traditional RAG Approach
![Vanilla RAG System](https://github.com/user-attachments/assets/vanilla-rag-diagram)

*Traditional RAG: Simple retrieval → augmentation → generation pipeline*

#### Our Agentic RAG Implementation  
![Agentic RAG System](https://github.com/user-attachments/assets/agentic-rag-diagram)

*Agentic RAG: Intelligent agents with reasoning loops and dynamic knowledge lookup*

### Agent Architecture Overview

#### 🎯 Thinking Agent (DeepSeek-Controlled)
- **Role**: Complex reasoning and analytical thinking
- **Capabilities**:
  - Think-Loop mechanism for iterative problem solving
  - Advanced mathematical and scientific reasoning
  - IB-specific curriculum analysis
  - Extended Essay and TOK guidance

#### 📚 RAGAgent (Qwen-Controlled)  
- **Role**: Knowledge retrieval and information synthesis
- **Capabilities**:
  - Reason-Loop for contextual understanding
  - Document retrieval from ChromaDB
  - Multi-source information integration
  - Subject-specific content expertise

---

### Agent Coordination Strategy

#### Inter-Agent Communication
- **Knowledge Sharing**: RAGAgent provides context to FastAgent
- **Insight Integration**: FastAgent reasoning enhances RAG responses  
- **Iterative Refinement**: Agents can request additional information from each other


---

## 🚧 Challenges & Solutions

### Challenge 1: Response Time Bottleneck ⏰

#### The Problem
Users experienced significant delays (30-60 seconds) when querying the system, particularly for complex multi-agent interactions.

#### Potential Root Causes
- **Model Processing Overhead**: DeepSeek's thinking loops require substantial computation
- **Time to Display**: Agents waited for complete responses before proceeding
- **Vector Database Queries**: ChromaDB similarity searches added latency
- **Context Loading**: Large IB document contexts slowed processing
- **Network API Calls**: Multiple calls to different language model endpoints

#### Solutions Implemented/Planned
```
- Implemented response streaming for immediate feedback
- Parallel agent processing where possible
- ChromaDB query optimization and caching
- Context window management and chunking
- Finetuning the thinking model and sidestepping a lot of the RAG files
```

### Challenge 2: DeepSeek Model Fine-tuning Failure 🎯

#### The Original Vision
Fine-tune DeepSeek model with IB-specific educational examples and marked rubrics to create a specialized IB reasoning model that could:
- Better understand IB assessment criteria
- Provide more accurate IA feedback
- Generate IB-specific study guidance

#### Technical Obstacles Encountered

**MLX Framework Compatibility Crisis**
- **Setup Swtich**: Switch From Windows To Mac
- **Dataset Format Issues**: Custom IB dataset didn't match MLX input requirements
- **Dependency Hell**: Multiple version conflicts between MLX, CUDA, and Python packages

**Current Alternative Strategy**
- Enhanced prompt engineering for IB-specific responses
- Context injection with IB rubrics and examples

---

## 📊 Project Achievements & Impact

### Technical Accomplishments
- ✅ **Working Multi-Agent System**: Functional agentic RAG implementation
- ✅ **IB-Specific Intelligence**: Tailored responses for IB curriculum
- ✅ **Document Processing**: PDF upload and analysis for IA assessment
- ✅ **Scalable Architecture**: Modular design supporting future enhancements
- ✅ **Real-time Interface**: Responsive web application with streaming

### Educational Impact
- 🎓 **Personalized Learning**: Adaptive responses based on IB requirements
- 📝 **IA Support**: Automated feedback on Internal Assessments  
- 🔍 **Curriculum Coverage**: Support across all IB subject groups
- 💡 **Critical Thinking**: DeepSeek-powered reasoning assistance

### Performance Metrics
```
Response Accuracy: High for IB-specific queries
Agent Coordination: Successful multi-agent collaboration
User Experience: Intuitive interface with real-time feedback
System Scalability: Modular architecture ready for expansion
```

---

## 🔮 Future Enhancements

- **Response Time Optimization**: Implement advanced caching strategies
- **Agent Specialization**: Subject-specific agent variants
- **Enhanced IA Assessment**: More detailed rubric-based feedback
- **Adding ALl Subjects**: Supporting ALl subjects

---

**Thank you for your attention!**

*Ready to discuss the technical implementation, agent architecture, and the future of AI in education.*

---



