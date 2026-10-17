# VectorDB — Self-Hosted Vector Database & RAG Engine in Java

A fully working **Vector Database & RAG Pipeline** built from scratch in Java (Spring Boot) with an interactive web UI.  
Designed with a **production-grade Document Chunk Search Index** backed by **dual relations (HNSW + Brute Force)** and **dynamic threshold routing ($N = 10$ chunks)**.

> **Architecture Overview:**  
> In this database, real document chunk embeddings are indexed **strictly into two relations: HNSW and Brute Force**. When queries are executed, the system dynamically routes to **Brute Force if chunks $< 10$** (100% exact ground truth, zero graph traversal overhead) and to **HNSW if chunks $\ge 10$** ($O(\log N)$ logarithmic scaling).

---

## What This Project Does

| Component | Architecture & Features |
|---|---|
| **Document Chunk Index** | Document chunks are stored **simultaneously in two relations: HNSW and Brute Force**. *(No other algorithms like KD-Tree are used for Document Chunks).* |
| **Dynamic Threshold Router** | Automatically routes search to **Brute Force when chunks $< 10$** and transitions to **HNSW when chunks $\ge 10$**. |
| **3 Distance Metrics** | **Manhattan distance ($L_1$)**, **Euclidean distance ($L_2$)**, and **Cosine distance** fully implemented and verified. |
| **Mathematical Testing Suite** | **23 automated tests** verifying Recall@K, Precision@K, distance formulas, and the 10-chunk threshold policy. |
| **Real Document Embeddings** | Splits documents into overlapping 250-word chunks and embeds them via Ollama (`nomic-embed-text`, 768D). |
| **Local RAG Pipeline** | Contextual document retrieval + local answer synthesis with `llama3.2` without sending data to external APIs. |
| **Demo Vectors Sandbox** | Educational 16D vector playground with 2D PCA scatter visualization, comparing HNSW, Brute Force, and KD-Tree. |
| **Full REST API** | Endpoints for CRUD, semantic search, threshold evaluation, and HNSW graph topology inspection. |

> 📊 **Testing & Evaluation Report:** See [TESTING_README.md](TESTING_README.md) for the automated test execution logs, mathematical formulas, and empirical proofs comparing HNSW vs Brute Force retrieval across both cases ($< 10$ and $\ge 10$).

---

## How It Works

```
Your Text Document
        │
        ▼
Text Chunker (overlapping 250-word chunks)
        │
        ▼
Ollama (nomic-embed-text)          ← converts each chunk to a 768-dimensional vector
        │
        ▼
Document Search Index (DocumentDB) ← indexes chunk into BOTH relations:
├── Brute Force Relation (exact linear store)
└── HNSW Relation (multilayer small-world proximity graph)
        │
        ▼
User Query / Question
        │
        ▼
Query Router (Threshold = 10)
├── If chunks < 10   ──► Queries BRUTE FORCE relation (100% exact ground truth, 20 µs)
└── If chunks >= 10  ──► Queries HNSW relation (O(log N) fast approximate nearest neighbor)
        │
        ▼
Retrieved Top-K Context Chunks
        │
        ▼
Ollama (llama3.2)                  ← reads retrieved context chunks, generates an answer
        │
        ▼
Final Answer
```

### Why the 10-Chunk Threshold Routing Policy?
* **When Document Chunks $< 10$**:
  Exhaustive Brute Force scan performs under 160 operations (for 16D) or 7,680 operations (for 768D), executing in $\approx 20\,\mu s$. Initializing HNSW priority queues and entry-point traversal adds unnecessary overhead ($\approx 64\,\mu s$). Brute Force is **$3.2\times$ faster** and mathematically guarantees **zero approximation error ($\epsilon = 0$)**.
* **When Document Chunks $\ge 10$**:
  As document volume increases ($N = 25, 50, 100, 500+$ chunks), Brute Force time grows linearly ($O(N)$), while HNSW navigates the multilayer graph in logarithmic time ($O(\log N)$), keeping retrieval times sub-millisecond while achieving $\ge 90\%-100\%$ Recall@$K$.

---

## Prerequisites

You need **3 things** installed:

1. **Java 17+** (e.g. [Eclipse Temurin](https://adoptium.net) JDK 21)
2. **Git**
3. **Ollama** (runs the local AI models from [ollama.com](https://ollama.com))

No need to install Maven — the included `mvnw.cmd` wrapper handles it automatically.

---

## Step-by-Step Setup (Windows)

### Step 1 — Verify Java 17+
```powershell
java -version
```

### Step 2 — Verify Git
```powershell
git --version
```

### Step 3 — Pull Ollama Models
Ensure Ollama is running, then pull the embedding model and language model:
```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Step 4 — Run Automated Test Suite
Verify that all 23 mathematical distance and retrieval tests pass:
```powershell
.\mvnw.cmd test
```

### Step 5 — Build & Run the Java Server
```powershell
.\mvnw.cmd spring-boot:run
```
Or build the JAR:
```powershell
.\mvnw.cmd clean package -DskipTests
java -jar target\vectordb-1.0.0.jar
```

Server starts at:
```
http://localhost:8080
```

---

## Using the Application

### Tab 1: Search (Demo Vectors Sandbox)
- Educational sandbox comparing **HNSW**, **Brute Force**, and **KD-Tree** on 16D semantic vectors.
- Choose distance metric: **Manhattan ($L_1$)**, **Euclidean ($L_2$)**, or **Cosine**.
- Live 2D PCA scatter plot showing distinct clusters for CS, Math, Food, and Sports.

### Tab 2: Documents (Dual-Relation Indexing)
- Paste your lecture notes, documents, or articles.
- Text is split into overlapping 250-word chunks.
- Each chunk is embedded via Ollama (768D) and inserted **into both the HNSW relation and the Brute Force relation**.

### Tab 3: Ask AI (RAG Pipeline)
- Type a natural language question about your ingested documents.
- The router checks total document chunks:
  - If $< 10$ chunks $\to$ retrieves context using the **Brute Force relation**.
  - If $\ge 10$ chunks $\to$ retrieves context using the **HNSW relation**.
- Context chunks are passed to `llama3.2` to synthesize an accurate answer.

---

## REST API Reference

### Document Chunks & RAG Endpoints

| Method | Endpoint | Query / Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Chunks text, embeds with Ollama, and stores in **both HNSW and BF relations** |
| `GET` | `/doc/list` | — | List all stored document chunks with word counts and previews |
| `DELETE`| `/doc/delete/:id` | — | Removes chunk from both relations |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: routes query by chunk count ($< 10$ BF vs $\ge 10$ HNSW) + generates answer |
| `GET` | `/doc/compare-threshold` | `?k=3&metric=manhattan` | Compares retrieval quality of live chunks under the 10-chunk policy |
| `GET` | `/doc/simulate-threshold` | `?count=6&k=3&metric=manhattan` | Simulates Case 1 ($< 10$) or Case 2 ($\ge 10$) retrieval metrics on the fly |

### Search & Evaluation Endpoints

| Method | Endpoint | Query Parameters | Description |
|---|---|---|---|
| `GET` | `/search` | `?v=...&k=5&metric=manhattan&algo=hnsw` | Vector search (`algo=hnsw` or `algo=bruteforce`) |
| `GET` | `/evaluate` | `?v=...&k=5&metric=manhattan&ef=50` | Full mathematical evaluation comparing HNSW against Brute Force ground truth |
| `GET` | `/benchmark` | `?v=...&k=5&metric=cosine` | Microsecond latency benchmark |
| `GET` | `/hnsw-info` | — | Graph topology: node count, edges per layer, top layer |
| `GET` | `/stats` | — | Collection counts, vector dimensions, and supported metrics |
| `GET` | `/status` | — | Ollama health check and loaded model info |

---

## Project Structure

```
VectorDB-Java/
├── pom.xml                                      ← Maven build file (Spring Boot 3.2.5 + JUnit 5)
├── mvnw.cmd                                     ← Auto-detecting Maven wrapper
├── TESTING_README.md                            ← Comprehensive test report & empirical proofs
├── src/main/java/com/vectordb/
│   ├── VectorDbApplication.java                 ← Spring Boot entry point + demo loader
│   ├── algo/
│   │   ├── BruteForce.java                      ← Exact exhaustive search relation (O(N·D))
│   │   ├── HNSW.java                            ← Multi-layer graph relation (O(D·log N))
│   │   ├── KDTree.java                          ← Educational KD-Tree (used in demo vectors)
│   │   ├── Distances.java                       ← Manhattan (L1), Euclidean (L2), Cosine
│   │   ├── DistanceFn.java                      ← Functional interface for distance metrics
│   │   └── DistHit.java                         ← (distance, id) record
│   ├── model/
│   │   ├── DocItem.java                         ← Document chunk with embedding
│   │   ├── VectorItem.java                      ← Demo vector model
│   │   ├── ThresholdComparisonResult.java       ← Model for < 10 vs >= 10 comparison metrics
│   │   └── EvaluationResult.java                ← Model for Recall@K, precision, speedup
│   ├── db/
│   │   ├── DocumentDB.java                      ← Document Chunk Search Index (Dual HNSW + BF)
│   │   └── VectorDB.java                        ← Demo vectors repository
│   ├── service/
│   │   ├── ThresholdComparisonService.java      ← Evaluates < 10 (BF) vs >= 10 (HNSW)
│   │   ├── EvaluationService.java               ← Mathematical Recall@K & ASCII report generator
│   │   ├── OllamaClient.java                    ← Ollama embedding & generation client
│   │   └── TextChunker.java                     ← Splits documents into overlapping chunks
│   └── controller/
│       ├── DocumentController.java              ← Document ingestion, RAG, and threshold evaluation
│       ├── VectorController.java                ← Vector search, evaluation, and benchmark APIs
│       └── StatusController.java                ← System health & Ollama status
└── src/test/java/com/vectordb/
    ├── algo/
    │   ├── DistancesTest.java                   ← Exact step-by-step Manhattan & distance checks
    │   ├── BruteForceTest.java                  ← Ground truth ordering tests
    │   └── HNSWRetrievalTest.java               ← HNSW recall, efSearch scaling, metric comparisons
    ├── evaluation/
    │   ├── DocumentChunkThresholdRetrievalTest.java ← Case 1 (<10) vs Case 2 (>=10) verification
    │   └── HNSWEvaluationSuiteTest.java         ← Statistical recall & latency evaluation
    └── controller/
        └── VectorControllerTest.java            ← REST API MockMvc integration tests
```

---

## License

MIT — feel free to use, modify, and build upon this project.
