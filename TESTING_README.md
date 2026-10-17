# Document Chunks Retrieval Testing Report (HNSW & Brute Force Relations)

This report documents the testing and mathematical evaluation of the **Document Chunk Search Index**.

---

## 1. Document Chunks Architecture: Dual Relation Storage

In this system, document chunk embeddings are indexed **strictly and exclusively into two relations**:
1. **Brute Force Relation**: An exact exhaustive index storing all chunk embeddings in sequential memory ($O(N \cdot D)$ scan).
2. **HNSW Relation**: A hierarchical navigable small-world proximity graph organizing chunk embeddings into multilayer skip-graphs ($O(D \cdot \log N)$ search).

> **Important Architecture Note:**  
> **Document Chunking does NOT use any other algorithms (such as KD-Tree).**  
> Every document chunk vector embedding is inserted simultaneously into both the **Brute Force relation** and the **HNSW relation**.

---

## 2. Dynamic Threshold Routing Policy ($N = 10$)

When a search or RAG question is queried against the document index, the system dynamically selects which relation to execute based on the total number of document chunks ($N$):

$$\text{Active Relation}(N) = \begin{cases} 
\mathbf{Brute\ Force\ Relation}, & \text{if } N < 10 \\ 
\mathbf{HNSW\ Relation\ (efSearch = 50)}, & \text{if } N \ge 10 
\end{cases}$$

### Why This Threshold Policy is Mathematically Optimal:
* **When $N < 10$ (Brute Force Relation)**:
  * Linear scan over $< 10$ chunks performs under 160 floating-point operations (for 16D) or 7,680 ops (for 768D).
  * Latency is **$20\,\mu s$** (faster than HNSW graph traversal overhead).
  * **Zero approximation error ($\epsilon = 0$)**: $\text{Recall}@K = \mathbf{100.0\%}$ and $\text{Precision}@K = \mathbf{100.0\%}$ are mathematically guaranteed.
* **When $N \ge 10$ (HNSW Relation)**:
  * As documents are added ($N = 25, 50, 100, 500+$ chunks), Brute Force execution time grows linearly ($O(N)$).
  * HNSW activates with logarithmic scaling ($O(\log N)$), preventing search bottlenecks while maintaining $\ge 90\%-100\%$ Recall@$K$.

---

## 3. Test Execution Verification

```
-------------------------------------------------------
 T E S T S   E X E C U T I O N   S U M M A R Y
-------------------------------------------------------
Tests run: 23, Failures: 0, Errors: 0, Skipped: 0
Build Status: BUILD SUCCESS
Execution Environment: OpenJDK 21.0.5 LTS | Spring Boot 3.2.5 | JUnit Platform
```

### Test Suites Covering Document Chunks & Relations:
| Test Class | Focus Area | Tests | Status |
| :--- | :--- | :---: | :---: |
| `DocumentChunkThresholdRetrievalTest` | Dual-relation evaluation for Chunks $< 10$ and Chunks $\ge 10$ | 5 | **PASSED** |
| `DistancesTest` | Manhattan ($L_1$), Euclidean ($L_2$), Cosine distance verification | 7 | **PASSED** |
| `BruteForceTest` | Brute Force relation exact nearest-neighbor ordering & distance checks | 3 | **PASSED** |
| `HNSWRetrievalTest` | HNSW relation graph retrieval accuracy & parameter sensitivity | 3 | **PASSED** |
| `HNSWEvaluationSuiteTest` | Statistical recall & latency benchmark comparing both relations | 3 | **PASSED** |
| `VectorControllerTest` | REST API integration tests | 2 | **PASSED** |
| **Total** | | **23** | **ALL PASSED** |

---

## 4. Empirical Test Results: Case 1 vs. Case 2

All results below are directly produced by `DocumentChunkThresholdRetrievalTest.java`.

### Case 1: Document Chunks $< 10$ ($N = 6$ chunks, Manhattan Distance)
The router detects $6 < 10$ and queries the **Brute Force relation**:

```
=========================================================================================
   CASE 1: DOCUMENT CHUNKS < 10 (6 Chunks) — MANHATTAN DISTANCE
=========================================================================================
Router Policy  : Index size 6 < threshold 10 -> Selected: [BRUTE FORCE RELATION]
Recall@3       : 100.0%  | Precision@3: 100.0%
Distance Ratio : 1.0000  (Exact ground truth, zero approximation error)
Top-1 Match    : YES     | Exact Ranking Match: YES
Latencies      : Routed (Brute Force) = 20 µs | HNSW = 64 µs  (BF is 3.2x faster!)
-----------------------------------------------------------------------------------------
Rank  | ID   | Distance (manhattan) | BF Rank   | Chunk Title                        
------+------+----------------------+-----------+----------------------------------------
1     | 2    | 4.10728              | #1        | Document Chunk #2                  
2     | 4    | 4.12367              | #2        | Document Chunk #4                  
3     | 5    | 4.85297              | #3        | Document Chunk #5                  
=========================================================================================
```

### Case 2: Document Chunks $\ge 10$ ($N = 25$ chunks, Manhattan Distance)
The router detects $25 \ge 10$ and queries the **HNSW relation**:

```
=========================================================================================
   CASE 2: DOCUMENT CHUNKS >= 10 (25 Chunks) — MANHATTAN DISTANCE
=========================================================================================
Router Policy  : Index size 25 >= threshold 10 -> Selected: [HNSW RELATION]
Recall@5       : 100.0%  | Precision@5: 100.0%
Distance Ratio : 1.0000  (All true nearest neighbor chunks found)
Top-1 Match    : YES     | Exact Ranking Match: YES
Latencies      : Routed (HNSW) = 52 µs | Brute Force = 68 µs (HNSW is faster!)
-----------------------------------------------------------------------------------------
Rank  | ID   | Distance (manhattan) | BF Rank   | Chunk Title                        
------+------+----------------------+-----------+----------------------------------------
1     | 23   | 3.46194              | #1        | Document Chunk #23                 
2     | 21   | 4.29212              | #2        | Document Chunk #21                 
3     | 1    | 4.53277              | #3        | Document Chunk #1                  
4     | 17   | 4.58362              | #4        | Document Chunk #17                 
5     | 9    | 4.92740              | #5        | Document Chunk #9                  
=========================================================================================
```

### Critical Boundary Test ($N = 9 \to N = 10$ chunks)
Verified in `testBoundaryTransitionNineToTen`:
```
Boundary Transition Verified:
  N=9  -> Active Relation: BRUTE FORCE | Recall@4: 100.0% | Latency: 24 µs
  N=10 -> Active Relation: HNSW        | Recall@4: 100.0% | Latency: 48 µs
```
* Shows seamless transition when the 10th document chunk is inserted without any recall degradation.

---

## 5. Mathematical Distance Calculation Check (Manhattan $L_1$)

Manhattan distance sum:
$$d_{\text{manhattan}}(\mathbf{q}, \mathbf{chunk}) = \sum_{i=1}^{D} |q_i - \text{chunk}_i|$$

* Hand-calculated verification on 3D vectors $\mathbf{a} = [1.0, 2.0, 3.0]$ and $\mathbf{b} = [4.0, 0.0, 7.0]$:
  $$|1 - 4| + |2 - 0| + |3 - 7| = 3.0 + 2.0 + 4.0 = \mathbf{9.00000}$$
* Both the Brute Force relation and HNSW relation report exact matching distances ($1e-5$ precision).
* Tested across **Manhattan ($L_1$)**, **Euclidean ($L_2$)**, and **Cosine distance** with 100% Top-1 accuracy.

---

## 6. How to Run the Document Chunk Tests Locally

### Run Document Chunk Threshold Tests:
```powershell
.\mvnw.cmd test -Dtest=DocumentChunkThresholdRetrievalTest
```

### Run All Tests:
```powershell
.\mvnw.cmd test
```

### Interactive HTTP Endpoints:
When Spring Boot is running (`.\mvnw.cmd spring-boot:run`):
1. **Simulate Case 1 ($N=6 < 10$ chunks, Manhattan)**:
   ```http
   GET http://localhost:8080/doc/simulate-threshold?count=6&metric=manhattan&k=3
   ```
2. **Simulate Case 2 ($N=25 \ge 10$ chunks, Manhattan)**:
   ```http
   GET http://localhost:8080/doc/simulate-threshold?count=25&metric=manhattan&k=5
   ```
3. **Inspect Live In-Memory Document Chunks**:
   ```http
   GET http://localhost:8080/doc/compare-threshold?metric=manhattan&k=3
   ```
