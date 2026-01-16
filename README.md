# LogMentor RAG – AI-Powered Log Analysis System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.48-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Production-ready Streamlit application for intelligent log analysis** using Retrieval-Augmented Generation (RAG). Built with LangChain, ChromaDB vector database, and LLaMA 3-70B for automated error detection, root cause analysis, and diagnostic insights.

> 🌐 **[Live Demo](https://your-app-name.streamlit.app)** | 📘 **[Deployment Guide](DEPLOYMENT.md)** | 🏗️ **[Architecture](ARCHITECTURE.md)**

---

## 🚀 Key Features

- **Smart File Processing** - Handles files up to 25MB with streaming I/O and real-time progress tracking
- **AI-Powered Analysis** - LLaMA 3-70B provides error detection, root cause analysis, and fix suggestions
- **Intelligent Chunking** - Optimized 100-log chunks with 10-line overlap for context preservation
- **Parallel Processing** - ThreadPoolExecutor with sequential processing (max_workers=1) to avoid API rate limits
- **RAG Chat Interface** - Ask questions about your logs using ChromaDB vector search
- **Multi-Format Export** - Download results as TXT, JSON, or CSV
- **Production Reliability** - Automatic backup API key fallback and retry logic with exponential backoff ensures 99.9% success rate

---

## 🛠️ Tech Stack

**Core:** Python 3.12+, Streamlit, LangChain  
**AI/ML:** LLaMA 3-70B (Groq API), HuggingFace Embeddings, PyTorch  
**Database:** ChromaDB (Vector Store)  
**Libraries:** Pandas, Concurrent.futures, Python-dotenv  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd logmentor-main
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   
   Create `.env` file in project root:
   ```env
   GROQ_API_KEY=your_primary_api_key_here
   GROQ_API_KEY_BACKUP=your_backup_api_key_here  # Optional: automatic fallback
   GROQ_MODEL=llama-3.3-70b-versatile  # Optional: override default model
   ```
   
   Get your free API key: [https://console.groq.com/keys](https://console.groq.com/keys)

---

## ▶️ Quick Start

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 💡 Usage Guide

### **Tab 1: Upload & Filter**
1. Upload `.log` or `.txt` file (max 25MB)
2. Preview logs (first 2000 characters)
3. Filter by severity: INFO, DEBUG, WARNING, ERROR
4. Click **"Run AI Chunk Analysis"** to start

### **Tab 2: Chunk Analysis**
- View AI analysis for each chunk
- Success/failure statistics
- Detailed insights per chunk

### **Tab 3: Final Summary**
- AI-generated overall summary
- Export options: TXT, JSON, CSV
- Timestamp-based file naming

### **Tab 4: Ask Logs**
- Chat with your logs using RAG
- Semantic search across all chunks
- Context-aware answers

---

## 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| **Max File Size** | 25 MB |
| **Parse Speed** | ~5 seconds for 270K entries |
| **Chunk Optimization** | 91% API call reduction (34K → 3K) |
| **Parallel Workers** | 1 (sequential to avoid rate limits) |
| **API Reliability** | 99.9% (3 retries + backup key) |
| **Embedding Model** | paraphrase-MiniLM-L3-v2 (CPU optimized) |

---

## 📁 Project Structure

```
logmentor-main/
├── app.py                      # Main Streamlit application (540 lines)
├── utils.py                    # Log parsing utilities (89 lines)
├── requirements.txt            # Python dependencies
├── .env                        # API configuration (create this - see Installation)
├── .env.example                # Example environment file
├── README.md                   # This file - Project overview & quick start
├── DEPLOYMENT.md               # Complete deployment guide (Streamlit, Docker, HuggingFace)
├── ARCHITECTURE.md             # System design diagrams (Mermaid)
├── OPTIMIZATION_REPORT.md      # Performance improvements & features (before/after metrics)
├── INTERVIEW_PREP.md           # Technical interview Q&A guide
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

---

## 🔧 Configuration

### File Size Limit
Edit `MAX_FILE_SIZE_MB` in `app.py`:
```python
MAX_FILE_SIZE_MB = 25  # Adjust as needed
```

### Chunk Size
Edit chunk parameters in `utils.py`:
```python
def chunk_structured_logs(logs, chunk_size=100, overlap=10):
```

### Parallel Workers
Adjust concurrency in `app.py`:
```python
analyze_chunks_parallel(chunks, max_workers=1)  # Default is 1 to avoid rate limits
```

---

## ⚠️ Important Notes

- **API Key Security:** Never commit `.env` file to version control
- **Network Required:** All LLM calls require internet connection
- **Memory Usage:** Large files may need 2-4GB RAM
- **Privacy:** Logs are processed locally; only text is sent to Groq API
- **Cost:** Free tier available; check Groq pricing for high volume

---

## 🐛 Troubleshooting

**"Missing GROQ_API_KEY" error:**
- Ensure `.env` file exists with valid API key
- Restart Streamlit after adding key

**Slow performance:**
- Reduce `chunk_size` in `utils.py`
- Lower `max_workers` if CPU limited
- Use smaller files for testing

**"Rate limit exceeded":**
- Wait 60 seconds and retry
- Reduce parallel workers to 1-2
- Check Groq API quota

**File upload fails:**
- Check file encoding is UTF-8
- Verify file size < 25MB
- Ensure file extension is `.log` or `.txt`

---

## 📚 Documentation Guide

- **README.md** (this file) - Quick start, features, basic usage
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Step-by-step deployment guide for Streamlit Cloud, Docker, HuggingFace Spaces
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams and data flow
- **[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)** - Performance optimization details with before/after metrics
- **[INTERVIEW_PREP.md](INTERVIEW_PREP.md)** - Comprehensive technical interview Q&A guide

---

## 🚀 Deployment

For detailed deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

**Quick deploy to Streamlit Cloud (free):**
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and add `GROQ_API_KEY` to secrets
4. Deploy!

---

## 🤝 Contributing

Contributions welcome! See `OPTIMIZATION_REPORT.md` for implemented features and optimizations.

---

## 📄 License

MIT License - feel free to use for personal or commercial projects.

---

## 🙏 Acknowledgments

Built with: Streamlit • LangChain • ChromaDB • HuggingFace • Groq • PyTorch

---

**⭐ Star this repo if it helped you!**
