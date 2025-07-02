# Neural Consciousness Memory System Flow Chart

## Overview

The Neural Consciousness Chat System implements a sophisticated memory architecture that combines short-term memory (STM), long-term memory (LTM), embeddings, vector search, and metacognitive evaluation. This document visualizes the complete flow from memory initialization through storage, retrieval, consolidation, and response enhancement.

## System Architecture Flow

```mermaid
graph TB
    %% Initialization Phase
    subgraph init["System Initialization"]
        A[FastAPI Startup] --> B[Memory Provider Factory]
        B --> C{Backend Selection}
        C -->|SQLite| D[PersonalMemorySystem]
        C -->|Redis| E[RedisMemoryStore]
        D --> F[Redis Compat Layer<br/>Optional]
        E --> F
        F --> G[Memory System Ready]
        
        A --> H[Redis Client Init]
        H --> I[Memory API Init]
        I --> J[Embedding Service Init]
        J --> K[Model Manager<br/>GPU Resource]
        K --> L[Vector Indices<br/>STM & LTM]
        
        I --> M[Consolidation Worker]
        M --> N[Retrieval Engine]
    end
    
    %% Chat Request Processing
    subgraph chat["Chat Request Flow"]
        O[User Message] --> P[Chat Stream Endpoint]
        P --> Q[Query Classification]
        Q --> R{Intent Type}
        R -->|Conversation| S[Memory Retrieval]
        R -->|Web Search| T[Web Scraper]
        R -->|Crypto/Stocks| U[Trading APIs]
        R -->|Maps| V[Google Maps API]
        
        S --> W[Get Relevant Memories]
        W --> X[Vector Search<br/>Embedding Service]
        X --> Y[Hybrid Search<br/>Vector + Metadata]
        Y --> Z[Re-ranking<br/>Multiple Signals]
        
        T --> AA[Enriched Prompt]
        U --> AA
        V --> AA
        Z --> AA
        
        AA --> AB[Token Manager<br/>Context Optimization]
        AB --> AC[LLM Generation]
    end
    
    %% Memory Storage Flow
    subgraph storage["Memory Storage"]
        AC --> AD[Response Generated]
        AD --> AE[Store Conversation<br/>Background Task]
        AE --> AF[Echo Detection<br/>Similarity Check]
        AF --> AG{Is Echo?}
        AG -->|No| AH[Calculate Importance]
        AG -->|Yes| AI[Skip Assistant Memory]
        
        AH --> AJ[Memory API<br/>Create STM]
        AJ --> AK[Extract Tags]
        AK --> AL[Determine Circle]
        AL --> AM[Storage Rules<br/>Enhanced Importance]
        AM --> AN[Check Capacity]
        AN --> AO{Over Limit?}
        AO -->|Yes| AP[Prune Old STMs]
        AO -->|No| AQ[Store in Redis<br/>With TTL]
        AP --> AQ
        
        AQ --> AR[Update Embedding Queue]
        AR --> AS[Batch Embedding<br/>Processing]
        AS --> AT[Update Vector Index]
    end
    
    %% Consolidation Flow
    subgraph consolidation["Memory Consolidation"]
        AU[Consolidation Worker] --> AV{Trigger Type}
        AV -->|Inactivity| AW[30min Timer]
        AV -->|Periodic| AX[24hr Timer]
        AV -->|Manual| AY[User Request]
        
        AW --> AZ[Get All STMs]
        AX --> AZ
        AY --> AZ
        
        AZ --> BA[Group by Circle]
        BA --> BB[DBSCAN Clustering<br/>Embedding Similarity]
        BB --> BC[Create Candidates]
        BC --> BD[Extract Common Tags]
        BD --> BE[Generate Summary<br/>LLM or Simple]
        BE --> BF[Queue for Review]
        
        BF --> BG[User Decision]
        BG --> BH{Approved?}
        BH -->|Yes| BI[Promote to LTM]
        BH -->|No| BJ[Discard Candidate]
        
        BI --> BK[Delete Original STMs]
        BK --> BL[Store LTM<br/>No TTL]
    end
    
    %% Retrieval Flow
    subgraph retrieval["Memory Retrieval"]
        BM[Search Request] --> BN[Retrieval Engine]
        BN --> BO[Generate Query Embedding]
        BO --> BP[Parallel Search]
        BP --> BQ[STM Vector Search]
        BP --> BR[LTM Vector Search]
        
        BQ --> BS[Apply Filters<br/>Tags, Circles, Time]
        BR --> BS
        
        BS --> BT[Combine Results]
        BT --> BU[Re-rank Results]
        BU --> BV[Tag Score]
        BU --> BW[Circle Priority]
        BU --> BX[Recency/Decay]
        BU --> BY[Vector Similarity]
        
        BV --> BZ[Composite Score]
        BW --> BZ
        BX --> BZ
        BY --> BZ
        
        BZ --> CA[Extract Highlights]
        CA --> CB[Format Context<br/>For LLM]
    end
    
    %% Metacognitive Flow
    subgraph metacog["Metacognitive Evaluation"]
        CC[Generated Response] --> CD[Heuristic Evaluator]
        CD --> CE[Fast Quality Checks]
        CE --> CF[Clarity Score]
        CE --> CG[Completeness Score]
        CE --> CH[Relevance Score]
        CE --> CI[Coherence Score]
        
        CF --> CJ[Response Assessment]
        CG --> CJ
        CH --> CJ
        CI --> CJ
        
        CJ --> CK{Needs Improvement?}
        CK -->|Yes| CL[LLM Critic<br/>Deep Analysis]
        CK -->|No| CM[Return Response]
        
        CL --> CN[Improvement Suggestions]
        CN --> CO[Regenerate Response<br/>With Feedback]
        CO --> CC
    end
    
    %% System Components
    subgraph components["Core Components"]
        CP[Memory Provider<br/>Factory Pattern] --> CQ[Memory Systems<br/>SQLite/Redis]
        CR[Memory API<br/>CRUD Operations] --> CS[Storage Rules<br/>Importance/Pruning]
        CT[Embedding Service<br/>GPU Accelerated] --> CU[Vector Indices<br/>Redis Search]
        CV[Retrieval Engine<br/>Hybrid Search] --> CW[Re-ranking Logic]
        CX[Consolidation Worker<br/>Background Tasks] --> CY[DBSCAN Clustering]
        CZ[Metacognitive Engine<br/>Quality Assessment] --> DA[Improvement Loop]
    end
    
    %% Data Flow Connections
    G --> O
    AD --> CC
    CB --> AA
    CM --> DB[Final Response to User]
    CO --> AC
```

## Memory System Components

### 1. **Memory Provider Factory** (`memory_provider.py`)
- Switches between Redis and SQLite backends
- Provides compatibility layer for migration
- Configuration-based backend selection

### 2. **Memory API** (`memory/services/memory_api.py`)
- Core CRUD operations for STM and LTM
- Tag extraction and circle determination
- Capacity management and pruning
- Consolidation candidate queuing

### 3. **Embedding Service** (`memory/services/embedding_service.py`)
- GPU-accelerated embedding generation
- Batch processing with caching
- Vector index management (STM & LTM)
- Asynchronous queue processing
- Model management with resource locking

### 4. **Retrieval Engine** (`memory/services/retrieval_engine.py`)
- Hybrid search (vector + metadata)
- Multi-signal re-ranking algorithm
- Context formatting for LLM consumption
- Related memory discovery
- Time-based search capabilities

### 5. **Consolidation Worker** (`memory/services/consolidation_worker.py`)
- Activity monitoring (30-minute inactivity)
- Periodic consolidation (24-hour cycle)
- DBSCAN clustering for similarity grouping
- User-in-the-loop approval system
- Summary generation (LLM or heuristic)

### 6. **Storage Rules Manager** (`memory/services/storage_rules_manager.py`)
- Enhanced importance calculation
- Capacity limit enforcement
- Pruning decision logic
- User approval requirements

### 7. **Metacognitive Engine** (`metacognitive_engine.py`)
- Heuristic evaluation (System 1)
- LLM-based criticism (System 2)
- Quality assessment across dimensions
- Iterative improvement suggestions
- Response regeneration with feedback

## Memory Flow Stages

### Stage 1: Initialization
1. FastAPI startup triggers memory system initialization
2. Memory provider selects backend (SQLite or Redis)
3. Redis client establishes connection
4. Memory API, Embedding Service, and Retrieval Engine initialize
5. Vector indices created for STM and LTM
6. Consolidation worker starts background tasks

### Stage 2: Request Processing
1. User message arrives at chat-stream endpoint
2. Query classification determines intent
3. Memory retrieval searches for relevant context
4. Vector search finds similar memories
5. Re-ranking applies multiple signals
6. Context formatted and integrated into prompt

### Stage 3: Response Generation
1. Token manager optimizes context window
2. LLM generates response stream
3. Response stored as conversation memory
4. Echo detection prevents redundant storage
5. Importance calculation determines memory value
6. Background tasks handle storage asynchronously

### Stage 4: Memory Storage
1. STM created with extracted tags
2. Memory circle determined by content
3. Storage rules calculate enhanced importance
4. Capacity checks trigger pruning if needed
5. Memory stored in Redis with TTL
6. Embedding generation queued for batch processing

### Stage 5: Consolidation Process
1. Worker monitors for inactivity or periodic trigger
2. STMs grouped by memory circles
3. DBSCAN clusters similar memories
4. Candidates created with common tags
5. Summaries generated for user review
6. Approved candidates promoted to LTM

### Stage 6: Metacognitive Evaluation
1. Generated response assessed by heuristics
2. Quality scores calculated across dimensions
3. Low-scoring responses trigger LLM criticism
4. Improvement suggestions generated
5. Response regenerated with feedback
6. Iterative improvement until quality threshold met

## Key Features

### Hybrid Architecture
- Combines rule-based heuristics with LLM intelligence
- Fast path for simple operations
- Deep analysis for complex scenarios

### Scalable Design
- Asynchronous operations throughout
- Background task processing
- GPU resource management
- Connection pooling and caching

### User-Centric Memory
- Importance-based retention
- Context-aware retrieval
- User approval for consolidation
- Echo detection to avoid redundancy

### Quality Assurance
- Multi-dimensional assessment
- Iterative improvement
- Confidence scoring
- Weak area identification

## Performance Optimizations

1. **Batch Processing**: Embeddings generated in batches
2. **Caching**: TTL cache for embeddings
3. **Async Operations**: Non-blocking I/O throughout
4. **Resource Management**: GPU lock prevents contention
5. **Pruning**: Automatic capacity management
6. **Indices**: Vector indices for fast similarity search

## Configuration Points

- Memory backend selection (SQLite/Redis)
- TTL for short-term memories
- Capacity limits for STM/LTM
- Consolidation thresholds
- Embedding model selection
- Quality assessment thresholds

## Integration Points

1. **FastAPI**: Main web framework
2. **Redis**: Vector storage and caching
3. **SQLite**: Alternative memory backend
4. **LLM Server**: Persistent model service
5. **GPU Manager**: Resource allocation
6. **Prometheus**: Metrics collection