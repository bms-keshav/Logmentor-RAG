# LogMentor RAG – Architecture Diagrams

This document contains Mermaid diagrams for the system architecture, block view, and key sequences.

Note: To render Mermaid in VS Code, use the “Markdown Preview Mermaid Support” extension or open on a platform that supports Mermaid.

---

## 1) System Architecture

```mermaid
flowchart LR
  U[User] -->|Upload logs / Ask| UI[Streamlit UI (app.py)]

  UI -->|parse, normalize| LP[Log Processing (utils.py)]
  LP --> CH[Chunking]
  CH --> DOCS[LangChain Documents]

  DOCS -->|embed| EMB[HuggingFace Embeddings\n(paraphrase-MiniLM-L3-v2)]
  EMB -->|vectors| VS[(ChromaDB / SQLite)]

  %% Chunk analysis path
  UI -->|Run Analysis| LLM[Groq Llama3-70B (ChatGroq)]
  CH -. text chunks .-> LLM
  LLM -->|per-chunk insights| UI

  %% RAG Q&A path
  UI -->|Ask| RET[Retriever]
  VS --> RET
  RET -->|relevant chunks| LLM
  LLM -->|answer| UI
```

---

## 2) Block Diagram (Component View)

```mermaid
flowchart TB
  U[User] --> UI[Streamlit UI (app.py)]

  subgraph Ingestion & Prep
    LP[Log Processing (utils.py)\nstructure_logs + chunk_structured_logs]
  end

  subgraph Vector Pipeline
    EMB[HuggingFace Embeddings\n(paraphrase-MiniLM-L3-v2)]
    VS[(ChromaDB / SQLite)]
  end

  subgraph LLM Services
    LLM[Groq Llama3-70B\n(ChatGroq)]
  end

  UI --> LP --> DOCS[Chunks as Documents]
  DOCS --> EMB --> VS

  %% Chunk Analysis path
  DOCS -. text chunks .-> LLM
  LLM -->|Chunk insights| UI

  %% RAG Q&A path
  UI -->|Query| VS
  VS -->|Retrieve top-k chunks| UI
  UI -->|Prompt with context| LLM
  LLM -->|Answer| UI
```

---

## 3) Sequence – Chunk Analysis

```mermaid
sequenceDiagram
  participant U as User
  participant UI as Streamlit UI
  participant P as Log Processing (utils.py)
  participant L as LLM (Groq Llama3)

  U->>UI: Upload logs / Click "Run AI Chunk Analysis"
  UI->>P: structure_logs(raw) + chunk_structured_logs()
  loop For each chunk
    UI->>L: Prompt(chunk)
    L-->>UI: Summary, Errors, Root cause, Fix
  end
  UI-->>U: Show chunk-wise results
```

---

## 4) Sequence – RAG Q&A

```mermaid
sequenceDiagram
  participant U as User
  participant UI as Streamlit UI
  participant E as HF Embeddings
  participant V as ChromaDB
  participant R as Retriever
  participant L as LLM (Groq)

  UI->>E: Embed chunks
  E->>V: Upsert vectors
  U->>UI: Enter query
  UI->>R: Retrieve top-k from V (invoke)
  R-->>UI: Relevant chunks
  UI->>L: Prompt with context via safe_llm_invoke
  L-->>UI: Answer
  UI-->>U: Display
```
