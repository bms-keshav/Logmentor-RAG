Here are the **4 Architecture Diagrams** derived strictly from your codebase (`app.py` and `utils.py`). You can draw these on a whiteboard or use the Mermaid code provided to generate them digitally.

### **1. The RAG Data Pipeline (ETL & Inference)**

*This diagram visualizes the end-to-end flow implemented in `app.py`, from file upload to the final answer.*

```mermaid
flowchart TD
    subgraph Client_Layer
        User[User] -->|Uploads .log File| UI[Streamlit UI]
        User -->|Asks Question| UI
    end

    subgraph Data_Processing_Layer ["utils.py & app.py"]
        UI -->|Raw Text Stream| P1["structure_logs()"]
        P1 -->|Structured Data| P2["chunk_structured_logs()"]
        P2 -->|Text Chunks| EMB[HuggingFace Embeddings]
    end

    subgraph Storage_Layer ["ChromaDB"]
        EMB -->|Vectors| DB[("Chroma VectorStore")]
    end

    subgraph Inference_Layer ["Groq API"]
        DB -.->|Retrieve Top-3 Context| RAG[RAG Chain]
        UI -.->|User Query| RAG
        RAG -->|Prompt| LLM["ChatGroq (LLaMA 3)"]
        LLM -->|Answer| UI
    end

    classDef code fill:#f9f,stroke:#333,stroke-width:2px;
    class P1,P2,RAG code;

```

---

### **2. The Sliding Window Algorithm (Memory Model)**

*This visualizes the `chunk_structured_logs` function in `utils.py`. It is critical for explaining how you prevent context loss.*

```mermaid
gantt
    title Sliding Window Chunking (Overlap = 10)
    dateFormat X
    axisFormat %s
    
    section Log Stream
    Log Entries 0-200    :0, 200
    
    section Chunk 1
    Logs 0 to 100        :active, 0, 100
    
    section Chunk 2
    Overlap (90-100)     :crit, 90, 100
    Logs 100 to 190      :active, 100, 190
    
    section Chunk 3
    Overlap (180-190)    :crit, 180, 190
    Logs 190 to 200      :active, 190, 200

```

*Note for Interview:* Point to the red/critical sections and say: *"This overlap ensures that if an error stack trace starts at line 95 and ends at 105, it is fully captured in Chunk 2."*

---

### **3. The ThreadPool Execution Model (Concurrency)**

*This visualizes the `analyze_chunks_parallel` function in `app.py`. It shows how you process multiple chunks without blocking.*

```mermaid
sequenceDiagram
    participant Main as Main Thread (Streamlit)
    participant Pool as ThreadPoolExecutor
    participant Worker as Worker Threads
    participant API as Groq API

    Main->>Pool: submit(analyze_chunk, chunk_1)
    Main->>Pool: submit(analyze_chunk, chunk_2)
    Main->>Pool: submit(analyze_chunk, chunk_3)
    
    par Parallel Execution
        Pool->>Worker: Task 1
        Worker->>API: HTTP Request
        API-->>Worker: JSON Response
        
        Pool->>Worker: Task 2
        Worker->>API: HTTP Request
        API-->>Worker: JSON Response
    end
    
    Worker-->>Main: yield Future.result() (via as_completed)
    Main->>Main: Update Progress Bar

```

---

### **4. The Fault-Tolerant Retry Logic (Control Flow)**

*This visualizes the `safe_llm_invoke` function in `app.py`. It proves your code is production-ready.*

```mermaid
flowchart TD
    Start([Start Request]) --> Attempt{Attempt < 3?}
    Attempt -- No --> Fail([Raise Exception])
    Attempt -- Yes --> Primary{Use Backup Key?}
    
    Primary -- No --> Call1[Call API with PRIMARY Key]
    Primary -- Yes --> Call2[Call API with BACKUP Key]
    
    Call1 --> Success{Success?}
    Call2 --> Success
    
    Success -- Yes --> End([Return Result])
    Success -- No --> ErrType{Is Rate Limit?}
    
    ErrType -- Yes --> Switch[Set Use Backup = True]
    Switch --> Backoff
    
    ErrType -- No --> Backoff[Sleep: 4 * 2^attempt]
    Backoff --> Attempt

```

### **5. High-Level System Architecture**

```mermaid
flowchart LR
  subgraph Frontend
    U[User] -->|1. Upload logs / Ask| UI[Streamlit UI]
  end

  subgraph Processing
    UI -->|2. Parse & Chunk| LP[Log Processor]
    LP -->|3. Text Chunks| EMB[HuggingFace Embeddings]
  end

  subgraph Storage
    EMB -->|4. Vectors| VS[("ChromaDB")]
  end

  subgraph AI_Cloud
    UI -->|5. Retrieve Context| VS
    VS -.->|6. Top-k Chunks| UI
    UI -->|7. Prompt + Context| LLM["Groq API (LLaMA 3)"]
    LLM -->|8. Answer| UI
  end

  %% Changed fill to darker blue and text to white for readability
  classDef component fill:#0277bd,stroke:#01579b,stroke-width:2px,color:#ffffff;
  class UI,LP,EMB,VS,LLM component;
```
