package com.vectordb.evaluation;

import com.vectordb.algo.Distances;
import com.vectordb.model.EvaluationResult;
import com.vectordb.model.VectorItem;
import com.vectordb.service.EvaluationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Automated HNSW Evaluation Suite")
public class HNSWEvaluationSuiteTest {

    private EvaluationService evaluationService;
    private List<VectorItem> demoVectors;

    @BeforeEach
    void setUp() {
        evaluationService = new EvaluationService();
        demoVectors = createDemoDataset();
    }

    private List<VectorItem> createDemoDataset() {
        List<VectorItem> list = new ArrayList<>();
        list.add(new VectorItem(1, "Linked List: nodes connected by pointers", "cs",
                List.of(0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f)));
        list.add(new VectorItem(2, "Binary Search Tree: O(log n) search and insert", "cs",
                List.of(0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f)));
        list.add(new VectorItem(3, "Dynamic Programming: memoization overlapping subproblems", "cs",
                List.of(0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f)));
        list.add(new VectorItem(4, "Graph BFS and DFS: breadth and depth first traversal", "cs",
                List.of(0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f)));
        list.add(new VectorItem(5, "Hash Table: O(1) lookup with collision chaining", "cs",
                List.of(0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f)));

        list.add(new VectorItem(6, "Calculus: derivatives integrals and limits", "math",
                List.of(0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f)));
        list.add(new VectorItem(7, "Linear Algebra: matrices eigenvalues eigenvectors", "math",
                List.of(0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f)));
        list.add(new VectorItem(8, "Probability: distributions random variables Bayes theorem", "math",
                List.of(0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f)));
        list.add(new VectorItem(9, "Number Theory: primes modular arithmetic RSA cryptography", "math",
                List.of(0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f)));
        list.add(new VectorItem(10, "Combinatorics: permutations combinations generating functions", "math",
                List.of(0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f)));

        list.add(new VectorItem(11, "Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                List.of(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f)));
        list.add(new VectorItem(12, "Sushi: vinegared rice raw fish and nori rolls", "food",
                List.of(0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f)));
        list.add(new VectorItem(13, "Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                List.of(0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f)));
        list.add(new VectorItem(14, "Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                List.of(0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f)));
        list.add(new VectorItem(15, "Croissant: laminated pastry with buttery flaky layers", "food",
                List.of(0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f)));

        list.add(new VectorItem(16, "Basketball: fast-paced shooting dribbling slam dunks", "sports",
                List.of(0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f)));
        list.add(new VectorItem(17, "Football: tackles touchdowns field goals and strategy", "sports",
                List.of(0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f)));
        list.add(new VectorItem(18, "Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                List.of(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f)));
        list.add(new VectorItem(19, "Chess: openings endgames tactics strategic board game", "sports",
                List.of(0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f)));
        list.add(new VectorItem(20, "Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                List.of(0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f)));

        return list;
    }

    @Test
    @DisplayName("Evaluation 1: Manhattan Distance Retrieval on Semantic CS Query")
    void testManhattanEvaluationOnSemanticCsQuery() {
        // Query strongly oriented towards CS (dims 0..3 high)
        List<Float> csQuery = List.of(
                0.92f, 0.81f, 0.77f, 0.70f, 0.10f, 0.08f, 0.12f, 0.11f,
                0.05f, 0.06f, 0.07f, 0.08f, 0.06f, 0.09f, 0.07f, 0.08f
        );

        EvaluationResult result = evaluationService.evaluateDataset(
                demoVectors, csQuery, 5, "manhattan", 50, 16, 200
        );

        String report = evaluationService.formatReport(result);
        System.out.println(report);

        assertTrue(result.distanceCalculationsValid, "All Manhattan distance calculations must match exact formula");
        assertTrue(result.top1Match, "Top-1 retrieved item must match ground truth");
        assertEquals(1.0f, result.recallAtK, 1e-4f, "Recall@5 should be 100% on the semantic demo dataset");
        assertEquals("cs", result.hnswRetrieval.get(0).category, "Top hit category should be 'cs'");
    }

    @Test
    @DisplayName("Evaluation 2: Multi-Metric Comparison on Food Query (Manhattan vs Euclidean vs Cosine)")
    void testMultiMetricEvaluationComparison() {
        // Query strongly oriented towards Food (dims 8..11 high)
        List<Float> foodQuery = List.of(
                0.07f, 0.08f, 0.06f, 0.09f, 0.08f, 0.07f, 0.09f, 0.08f,
                0.88f, 0.85f, 0.80f, 0.75f, 0.07f, 0.08f, 0.06f, 0.09f
        );

        String[] metrics = {"manhattan", "euclidean", "cosine"};

        for (String metric : metrics) {
            EvaluationResult result = evaluationService.evaluateDataset(
                    demoVectors, foodQuery, 5, metric, 50, 16, 200
            );

            String report = evaluationService.formatReport(result);
            System.out.println(report);

            assertTrue(result.distanceCalculationsValid, "Distance calculations must be valid for " + metric);
            assertTrue(result.recallAtK >= 0.8f, "Recall@5 for " + metric + " should be at least 80%");
            assertEquals("food", result.hnswRetrieval.get(0).category, "Top hit for " + metric + " should be 'food'");
        }
    }

    @Test
    @DisplayName("Evaluation 3: Scaled Dataset (N=300) Manhattan Retrieval Precision & Recall across efSearch")
    void testScaledDatasetManhattanEvaluation() {
        Random rng = new Random(777);
        int dims = 16;
        int n = 300;

        List<VectorItem> scaledDataset = new ArrayList<>(n);
        for (int i = 1; i <= n; i++) {
            List<Float> emb = new ArrayList<>(dims);
            for (int d = 0; d < dims; d++) {
                emb.add(rng.nextFloat());
            }
            scaledDataset.add(new VectorItem(i, "scaled-item-" + i, "group-" + (i % 5), emb));
        }

        List<Float> query = new ArrayList<>(dims);
        for (int d = 0; d < dims; d++) query.add(rng.nextFloat());

        int[] efValues = {10, 30, 80};
        int k = 10;

        for (int ef : efValues) {
            EvaluationResult result = evaluationService.evaluateDataset(
                    scaledDataset, query, k, "manhattan", ef, 16, 200
            );

            System.out.printf(">>> Scaled N=%d | efSearch=%d | Recall@%d: %.1f%% | BF: %d µs | HNSW: %d µs%n",
                    n, ef, k, result.recallAtK * 100.0f, result.bruteforceLatencyUs, result.hnswLatencyUs);

            assertTrue(result.distanceCalculationsValid);
            assertTrue(result.recallAtK >= 0.7f, "Recall should be >= 70% even at low ef");
        }
    }
}
