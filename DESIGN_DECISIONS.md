# VectorDB — Architectural & Algorithmic Design Decisions

This document records the foundational architectural decisions made for the **VectorDB Document Search Engine**, explaining why specific mathematical distance metrics, data structures, and threshold routing rules were chosen.

---

## 1. Distance Metric: Why Cosine Distance Instead of Euclidean (Pythagoras) or Manhattan?

### The Core Problem: The "Document Length Trap"
In a real-world document search and RAG system, text chunks have varying lengths. 
Consider two document chunks:
* **Chunk A (Short Note, 8 words):** *"Calculus involves derivatives, integrals, and limits for functions."*
* **Chunk B (Detailed Chapter, 600 words):** An extensive discussion of calculus, repeating terms like *derivatives*, *integrals*, and *limits* dozens of times.

Both chunks describe the exact same concept (Calculus). However, when converted to dense vectors:
* Chunk A's vector coordinates have **smaller magnitudes**.
* Chunk B's vector coordinates have **much larger magnitudes** because frequency and depth scale up the vector length.

---

### Why Euclidean Distance (Pythagorean Theorem) Fails Here:
The Euclidean formula calculates the straight-line physical distance:
$$d_{\text{Euclidean}}(\mathbf{q}, \mathbf{c}) = \sqrt{\sum_{i=1}^{D} (q_i - c_i)^2}$$

Because Chunk B is much longer than Chunk A, the straight-line coordinate distance $\sqrt{\sum (q_i - c_i)^2}$ is **huge**. Euclidean distance misinterprets document length as a lack of semantic similarity.

---

### Why Manhattan Distance ($L_1$) Fails Here:
The Manhattan formula sums absolute coordinate gaps:
$$d_{\text{Manhattan}}(\mathbf{q}, \mathbf{c}) = \sum_{i=1}^{D} |q_i - c_i|$$

In high-dimensional embeddings (e.g. 768 dimensions), summing raw coordinate differences across hundreds of dimensions heavily penalizes longer documents and variations in magnitude, compounding the length error.

---

### Why Cosine Distance is the Standard for Text:
Cosine distance measures the **angle** between the query and document vectors, completely normalizing for document length:

$$\text{Cosine Similarity} = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\| \|\mathbf{c}\|} = \frac{\sum q_i c_i}{\sqrt{\sum q_i^2} \sqrt{\sum c_i^2}}$$

$$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$

```
                Chunk B (Long 600-word text)
                     /
                    /
                   /   Angle θ ≈ 0° 
                  /    Cosine Distance = 1 - cos(0°) = 0.0 (Perfect match!)
                 /
                / Chunk A (Short 8-word text)
               /
              • (Origin 0,0)
```

* **Outcome:** Even though Chunk B has larger raw coordinate numbers than Chunk A, both point in the **same semantic direction ($\theta \approx 0^\circ$)**. Cosine distance recognizes them as identical in meaning.

---

## 2. Search Engine Architecture: Dual Relations (HNSW + Brute Force)

Every ingested document chunk vector embedding is stored simultaneously in **two internal relations**:
1. **Brute Force Relation**: An exact, sequential vector list.
2. **HNSW Relation**: A multi-layer navigable small-world proximity graph.

### The Threshold = 10 Routing Policy:
$$\text{Active Search Relation}(N) = \begin{cases} 
\mathbf{Brute\ Force\ Relation}, & \text{if } N < 10 \\ 
\mathbf{HNSW\ Relation\ (efSearch = 50)}, & \text{if } N \ge 10 
\end{cases}$$

### Why Not Use HNSW for Small Datasets ($N < 10$)?
* For $N < 10$ chunks, Brute Force computes distance to all chunks in **$20\,\mu s$**.
* HNSW requires priority queues, visited tracking sets, and multi-layer descent, taking **$64\,\mu s$** on tiny datasets.
* Therefore, for $N < 10$, Brute Force is **$3.2\times$ faster** and mathematically guarantees **$100\%$ exact ground truth ($\epsilon = 0$)**.

### Why Not Use Brute Force for Larger Datasets ($N \ge 10$)?
* Brute Force is $O(N \cdot D)$ (linear). As documents scale ($N = 100, 1000+$), query time climbs linearly, bottlenecking the RAG pipeline.
* HNSW is $O(D \cdot \log N)$ (logarithmic). Graph traversal navigates directly to the nearest cluster, keeping query times flat and sub-millisecond.

---

## 3. What is Recall@K and Why Does It Matter?

In Approximate Nearest Neighbor (ANN) search:
* **$K$** is the number of results requested (e.g. $K=3$ for Top-3 context chunks in RAG).
* **Brute Force** represents the **Teacher's Answer Key** (exhaustive 100% ground truth).
* **HNSW** represents the **Fast Student** (navigating graph highways to approximate the top answers).

### Recall@$K$ Formula:
$$\text{Recall}@K = \frac{|\text{HNSW Top-}K \cap \text{BruteForce Top-}K|}{K}$$

* **Recall@3**: Checks if the Top-3 chunks selected by HNSW match the true Top-3 chunks found by Brute Force.
  * In RAG pipelines, LLMs typically take the **Top-3 chunks** as context. If Recall@3 is 100%, the LLM receives the exact same context as exhaustive Brute Force, but in a fraction of the time.
* **Recall@100**: Evaluates HNSW on a broader candidate pool (used in multi-stage retrieval / re-ranking systems).

---

## 4. How HNSW Works Mathematically

HNSW organizes vectors into a multi-layer graph inspired by skip-lists:

```
Layer 2 (Highway)     [Node 1] ──────────────────────────► [Node 9]
                          │                                    │
Layer 1 (Express)     [Node 1] ────────► [Node 5] ────────► [Node 9]
                          │                  │                 │
Layer 0 (Local/All)   [Node 1] ──► [Node 3] ─► [Node 5] ─► [Node 7] ─► [Node 9]
```

1. **Greedy Layer Traversal**: The search begins at the highest layer at an entry point node. It calculates cosine distance to all neighbors of that node. If a neighbor is closer to query $\mathbf{q}$, it hops to that neighbor.
2. **Dropping Layers**: When no neighbor in the current layer is closer to $\mathbf{q}$, it drops down one layer to explore a denser neighborhood.
3. **Layer 0 (Beam Search with `efSearch`)**: At the bottom layer containing all nodes, it runs a priority-queue beam search keeping the `efSearch` closest candidates, returning the final Top-$K$ items.

---

## 5. `efSearch` Variation Analysis: What Happens When You Change ef to 20, 50, or 100?

The `efSearch` parameter controls the size of the candidate pool on **Layer 0** before selecting the Top-$K$ (Top-3) chunks.

### Empirical Variation Benchmark Table:
| `efSearch` | Candidate Pool Size | Recall@3 (%) | Latency (µs) | Trade-Off Behavior |
| :---: | :---: | :---: | :---: | :--- |
| **`ef = 1`** | 1 candidate | **66.7%** | $33\,\mu s$ | 🔴 **Danger Zone**: Rushed search, missed 1 of the top 3 chunks! |
| **`ef = 3`** | 3 candidates | **66.7%** | $28\,\mu s$ | 🔴 **Danger Zone**: Candidate queue too small; gets stuck in local minima. |
| **`ef = 10`**| 10 candidates | **100.0%** | $46\,\mu s$ | 🟡 **Borderline**: Fast, but tight margin on complex data. |
| **`ef = 20`**| 20 candidates | **100.0%** | $58\,\mu s$ | 🟢 **Fast & Reliable**: Reaches 100% recall with low latency. |
| **`ef = 50`**| 50 candidates | **100.0%** | $112\,\mu s$| 🌟 **Production Sweet Spot**: 100% Recall guaranteed across all tests. |
| **`ef = 100`**| 100 candidates| **100.0%** | $154\,\mu s$| ⚪ **Diminishing Returns**: 0% extra accuracy gained, but consumes 40% more CPU. |
| **`ef = 200`**| 200 candidates| **100.0%** | $140-190\,\mu s$| ⚪ **Overkill**: Evaluates hundreds of extra neighbors with zero recall benefit. |

### Visual Graph:
See `efsearch_variation_graph.png` for the side-by-side plots:
* **Recall Curve**: Jumps from $66.7\%$ at $ef < 5$ to $100\%$ at $ef \ge 20$ (plateauing at $100\%$).
* **Latency Curve**: Rises from $30\,\mu s$ to $154\,\mu s$ as $ef$ increases.
* **Conclusion**: `ef = 50` is the optimal default for RAG because it guarantees $100\%$ Recall@3 while remaining well under a fraction of a millisecond.
