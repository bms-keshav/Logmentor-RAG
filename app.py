import sys
import sqlite3

sys.modules["sqlite3"] = sqlite3

import streamlit as st
from dotenv import load_dotenv
import os
import datetime
import pandas as pd
from utils import structure_logs, chunk_structured_logs
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor
import time

# Configure page to reduce white screen flash
st.set_page_config(
    page_title="LogMentor RAG",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Allow model to be configured via .env (so you can switch when a model is decommissioned)
# Default updated per Groq deprecations: https://console.groq.com/docs/deprecations
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Configuration
MAX_FILE_SIZE_MB = 25  # Maximum file size in MB
ALLOWED_EXTENSIONS = ['.log', '.txt']

# Init LLM + Embeddings
# Initialize ChatGroq with a model name coming from the environment so you can
# switch models without editing code when a model is deprecated.
# Cache LLM initialization to avoid recreating on every rerun
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

llm = get_llm()

# Auto-detect GPU for embeddings (faster if available)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Use lighter model for faster CPU embeddings (cached to avoid reloading)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": device}
    )

embedding = load_embeddings()


# Validate environment configuration
def validate_environment():
    """Ensure all required environment variables are set"""
    required_vars = {
        "GROQ_API_KEY": "Get from https://console.groq.com/keys"
    }
    
    missing = []
    for var, help_text in required_vars.items():
        if not os.getenv(var):
            missing.append(f"❌ {var}: {help_text}")
    
    if missing:
        st.error("### ⚙️ Configuration Error")
        for msg in missing:
            st.error(msg)
        st.info("Add missing variables to `.env` file and restart")
        st.stop()

# Run validation
validate_environment()


# Retry logic for LLM API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def safe_llm_invoke(llm, prompt):
    """
    Call LLM with automatic retry logic.
    Retries up to 3 times with exponential backoff if API fails.
    """
    try:
        logger.info("Calling LLM API...")
        result = llm.invoke(prompt)
        logger.info("LLM API call successful")
        return result
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise


def analyze_chunk(chunk_id, chunk, llm):
    """
    Analyze a single chunk (designed to run in thread pool).
    Returns dict with chunk_id, analysis, and status.
    """
    # chunk is already a formatted string from chunk_structured_logs
    prompt = f"""
You are an expert DevOps engineer analyzing server logs.

Chunk {chunk_id}:
{chunk}

Provide:
1. Summary of this chunk
2. Errors or warnings present
3. Root cause analysis (if errors exist)
4. Suggested fixes
"""
    
    try:
        result = safe_llm_invoke(llm, prompt)
        logger.info(f"✅ Chunk {chunk_id} analyzed successfully")
        return {
            "chunk_id": chunk_id,
            "analysis": result.content,
            "status": "success"
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Chunk {chunk_id} failed after 3 retries: {error_msg}")
        return {
            "chunk_id": chunk_id,
            "analysis": f"⚠️ Analysis failed after 3 attempts.\nError: {error_msg[:200]}...",
            "status": "failed"
        }


def analyze_chunks_parallel(chunks, llm, max_workers=3, progress_callback=None):
    """
    Analyze chunks in parallel using ThreadPoolExecutor.
    Returns list of results and statistics.
    
    Args:
        chunks: List of log chunks to analyze
        llm: LLM instance
        max_workers: Number of parallel workers
        progress_callback: Optional callback(completed, total) for progress updates
    """
    results = []
    total = len(chunks)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(analyze_chunk, i + 1, chunk, llm): i
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results as they complete
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed, total)
    
    # Sort by chunk_id to maintain order
    results.sort(key=lambda x: x["chunk_id"])
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    return results, successful, failed


# Initialize session state
if "all_chunk_insights" not in st.session_state:
    st.session_state.all_chunk_insights = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar
with st.sidebar:
    st.title("📘 How to Use")
    st.markdown("""
    1. Upload .log or .txt  
    2. Filter logs by severity  
    3. Run AI Chunk Analysis  
    4. Generate Summary or Ask Questions
    """)
    
    with st.expander("⚙️ Configuration"):
        gpu_status = "🚀 GPU Enabled" if device == "cuda" else "💻 CPU Mode (3x faster model)"
        st.code(f"""
Model: {GROQ_MODEL}
API Key: {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}
Max File Size: {MAX_FILE_SIZE_MB}MB
Embedding: paraphrase-MiniLM-L3-v2 (optimized for speed)
Device: {gpu_status}
        """)

# Main UI
st.markdown("<h1 style='text-align:center;'>🧠 LogMentor RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Structured Log Analysis with ChromaDB + LLM</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Upload & Filter", "🔍 Chunk Analysis", "🧠 Final Summary", "💬 Ask Logs"
])

# Tab 1 – Upload + Filter
with tab1:
    uploaded_file = st.file_uploader("📁 Upload log file", type=["txt", "log"])
    if uploaded_file:
        # Check if this is a new file (clear old state)
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.all_chunk_insights = []
            st.session_state.chunks = []
            st.session_state.vectorstore = None
            logger.info(f"New file uploaded: {uploaded_file.name}, clearing old state")
        
        # Validation 1: File size check
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large: {file_size_mb:.1f} MB (Maximum: {MAX_FILE_SIZE_MB} MB)")
            st.info("💡 Split your log file or adjust MAX_FILE_SIZE_MB in code for local use")
            st.stop()
        
        # Validation 2: File extension check
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            st.error(f"❌ Unsupported file type: {file_ext}")
            st.info(f"💡 Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
            st.stop()
        
        # Validation 3: Content validation with progress indicator
        try:
            with st.spinner("📖 Reading file..."):
                # Read file in chunks to avoid UI freeze on large files
                chunks = []
                chunk_size = 1024 * 1024  # 1MB chunks
                total_size = uploaded_file.size
                read_size = 0
                
                # Show progress for files > 5MB
                if total_size > 5 * 1024 * 1024:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        chunk = uploaded_file.read(chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk.decode("utf-8", errors="strict"))
                        read_size += len(chunk)
                        progress = read_size / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Reading: {read_size/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB ({progress*100:.0f}%)")
                    
                    progress_bar.empty()
                    status_text.empty()
                else:
                    # Small files - read directly
                    chunks.append(uploaded_file.read().decode("utf-8", errors="strict"))
                
                raw_text = "".join(chunks)
            
            if len(raw_text.strip()) == 0:
                st.error("❌ File is empty")
                st.stop()
                
        except UnicodeDecodeError:
            st.error("❌ File contains invalid UTF-8 encoding. Please check file encoding.")
            st.stop()
        
        # Show success with file info
        st.success(f"✅ Loaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        st.text_area("📄 Preview Logs", raw_text[:2000], height=300)

        # Log filtering
        level = st.selectbox("🔎 Filter by Level", ["All", "INFO", "DEBUG", "WARNING", "ERROR"])
        if level != "All":
            raw_text = "\n".join([line for line in raw_text.splitlines() if level in line])
            st.text_area("📄 Filtered Logs", raw_text[:2000], height=300)

        # Process logs with progress indicator
        with st.status("⚙️ Processing logs...", expanded=True) as status:
            status.write("📖 Parsing log entries...")
            structured_logs = structure_logs(raw_text)
            status.write(f"✅ Parsed {len(structured_logs)} log entries")
            
            status.write("✂️ Creating analysis chunks...")
            st.session_state.chunks = chunk_structured_logs(structured_logs)
            status.write(f"✅ Created {len(st.session_state.chunks)} chunks")
            
            status.update(label="✅ Processing complete!", state="complete")

        st.success(f"✅ Ready to analyze: {len(structured_logs)} log entries in {len(st.session_state.chunks)} chunks")

        # Run AI Chunk Analysis
        if st.button("🚀 Run AI Chunk Analysis"):
            st.session_state.all_chunk_insights.clear()
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            total_chunks = len(st.session_state.chunks)
            
            # Progress callback function
            def update_progress(completed, total):
                progress = completed / total
                progress_bar.progress(progress)
                status_text.text(f"⏳ Analyzing: {completed}/{total} chunks complete ({progress*100:.0f}%)")
            
            status_text.text(f"Starting parallel analysis of {total_chunks} chunks...")
            
            # Run parallel analysis with progress updates
            results, successful_chunks, failed_chunks = analyze_chunks_parallel(
                st.session_state.chunks, 
                llm,
                max_workers=3,  # Process 3 chunks at a time
                progress_callback=update_progress
            )
            
            # Store results
            st.session_state.all_chunk_insights = results
            
            # Calculate time
            elapsed_time = time.time() - start_time
            
            # Update progress to 100%
            progress_bar.progress(1.0)
            
            # Show final status
            if failed_chunks == 0:
                st.success(f"✅ Successfully analyzed all {successful_chunks} chunks in {elapsed_time:.1f}s!")
                st.balloons()  # Celebration animation!
            elif successful_chunks > 0:
                st.warning(f"⚠️ Analyzed {successful_chunks} chunks successfully, {failed_chunks} failed ({elapsed_time:.1f}s)")
            else:
                st.error("❌ All chunk analyses failed. Please check your API key and try again.")

# Tab 2 – Chunk-wise LLM Analysis
with tab2:
    st.subheader("🔍 Chunk-by-Chunk Analysis")
    if st.session_state.all_chunk_insights:
        # Show success/failure stats
        total = len(st.session_state.all_chunk_insights)
        successful = sum(1 for c in st.session_state.all_chunk_insights if c.get('status') == 'success')
        failed = total - successful
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", total)
        col2.metric("✅ Successful", successful)
        col3.metric("❌ Failed", failed)
        
        st.markdown("---")
        
        for insight in st.session_state.all_chunk_insights:
            chunk_id = insight["chunk_id"]
            status = insight.get("status", "unknown")
            
            # Color-code based on status
            if status == "success":
                st.success(f"**Chunk {chunk_id}** ✅")
            elif status == "failed":
                st.error(f"**Chunk {chunk_id}** ❌ (Analysis Failed)")
            else:
                st.info(f"**Chunk {chunk_id}**")
            
            st.markdown(insight["analysis"])
            st.markdown("---")
    else:
        st.info("No analysis yet. Go to 'Upload & Filter' tab and click 'Run Analysis'.")

# Tab 3 – Final Summary + Download
with tab3:
    if st.session_state.all_chunk_insights:
        # Extract analysis text from dictionaries
        combined = "\n\n".join([res["analysis"] for res in st.session_state.all_chunk_insights])
        final_prompt = f"Summarize all these chunk-wise log analyses:\n{combined}"
        try:
            final_result = llm.invoke(final_prompt)
            summary_text = final_result.content
        except Exception as e:
            st.error(f"LLM error while creating final summary: {e}\nCheck GROQ_MODEL in your .env and Groq deprecation docs.")
            summary_text = None

        if summary_text:
            st.success("📄 Final Summary")
            st.markdown(summary_text)

            # Export Options
            st.subheader("📥 Export Analysis")
            col1, col2, col3 = st.columns(3)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # TXT Export
            with col1:
                file_name_txt = f"log_summary_{timestamp}.txt"
                st.download_button(
                    "📄 Download TXT",
                    data=summary_text,
                    file_name=file_name_txt,
                    mime="text/plain"
                )
            
            # JSON Export
            with col2:
                import json
                json_data = {
                    "timestamp": timestamp,
                    "summary": summary_text,
                    "chunks_analyzed": len(st.session_state.all_chunk_insights),
                    "successful": sum(1 for c in st.session_state.all_chunk_insights if c.get('status') == 'success'),
                    "failed": sum(1 for c in st.session_state.all_chunk_insights if c.get('status') == 'failed'),
                    "chunk_details": st.session_state.all_chunk_insights
                }
                file_name_json = f"log_analysis_{timestamp}.json"
                st.download_button(
                    "📊 Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=file_name_json,
                    mime="application/json"
                )
            
            # CSV Export
            with col3:
                csv_data = "Chunk ID,Status,Analysis\n"
                for chunk in st.session_state.all_chunk_insights:
                    # Escape quotes and newlines for CSV
                    analysis_clean = chunk['analysis'].replace('"', '""').replace('\n', ' ')
                    csv_data += f"{chunk['chunk_id']},{chunk['status']},\"{analysis_clean}\"\n"
                
                file_name_csv = f"log_analysis_{timestamp}.csv"
                st.download_button(
                    "📊 Download CSV",
                    data=csv_data,
                    file_name=file_name_csv,
                    mime="text/csv"
                )
    else:
        st.info("No chunks analyzed. Please analyze logs first.")

# Tab 4 – Chat With Logs (RAG)
with tab4:
    if st.session_state.chunks:
        # Cache vectorstore to prevent memory leak
        if st.session_state.vectorstore is None:
            with st.spinner("📦 Embedding and indexing chunks (one-time setup)..."):
                documents = [Document(page_content=chunk) for chunk in st.session_state.chunks]
                st.session_state.vectorstore = Chroma.from_documents(documents, embedding)
                st.success("✅ Vector database created and cached!")
        else:
            st.info("💾 Using cached vector database (fast!)")
        
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        user_query = st.text_input("💬 Ask something about the logs")
        if user_query:
            with st.spinner("💡 Thinking..."):
                try:
                    response = qa_chain.run(user_query)
                except Exception as e:
                    st.error(f"Error during retrieval/LLM call: {e}\nIf this mentions a decommissioned model, update GROQ_MODEL in .env.")
                    response = None

                if response:
                    st.markdown("🧠 **Answer**")
                    st.markdown(response)
    else:
        st.info("No logs to query. Please upload and analyze first.")