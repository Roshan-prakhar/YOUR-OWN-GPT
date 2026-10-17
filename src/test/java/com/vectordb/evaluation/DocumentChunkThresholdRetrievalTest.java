package com.vectordb.evaluation;

import com.vectordb.algo.Distances;
import com.vectordb.db.DocumentDB;
import com.vectordb.model.DocItem;
import com.vectordb.model.ThresholdComparisonResult;
import com.vectordb.service.ThresholdComparisonService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Document Chunks Threshold Retrieval Quality Test Environment (< 10 vs >= 10)")
public class DocumentChunkThresholdRetrievalTest {

    private ThresholdComparisonService comparisonService;
    private static final int DIMS = 16;

    @BeforeEach
    void setUp() {
        comparisonService = new ThresholdComparisonService();
    }

    private List<DocItem> generateChunks(int count, long seed) {
        Random rng = new Random(seed);
        List<DocItem> list = new ArrayList<>(count);
        for (int i = 1; i <= count; i++) {
            List<Float> emb = new ArrayList<>(DIMS);
            for (int d = 0; d < DIMS; d++) {
                emb.add(rng.nextFloat());
            }
            list.add(new DocItem(i, "Document Chunk #" + i, "Text chunk body content for chunk " + i, emb));
        }
        return list;
    }

    private List<Float> generateQuery(long seed) {
        Random rng = new Random(seed);
        List<Float> q = new ArrayList<>(DIMS);
        for (int d = 0; d < DIMS; d++) q.add(rng.nextFloat());
        return q;
    }

    @Test
    @DisplayName("Case 1: Document chunks < 10 (N=6) - Router selects BruteForce with exact 100% recall")
    void testCase1UnderThresholdChunks() {
        int chunkCount = 6; // Less than 10
        List<DocItem> chunks = generateChunks(chunkCount, 42);
        List<Float> query = generateQuery(99);
        int k = 3;

        // Test with Manhattan Distance
        ThresholdComparisonResult result = comparisonService.evaluateDataset(
                chunks, query, k, "manhattan", 50
        );

        String report = comparisonService.formatReport(result);
        System.out.println(report);

        // Mathematical Assertions for Case 1 (N < 10):
        assertTrue(result.isUnderThreshold, "Should identify that chunk count is < 10");
        assertEquals("bruteforce", result.routedAlgorithm, "Router must select Brute Force when chunks < 10");
        assertEquals(1.0f, result.recallAtK, 1e-4f, "Brute Force must yield mathematically exact 100% Recall@K");
        assertEquals(1.0f, result.precisionAtK, 1e-4f, "Brute Force must yield mathematically exact 100% Precision@K");
        assertEquals(1.0000, result.distanceRatio, 1e-4, "Distance ratio must be exactly 1.0000 (zero approximation error)");
        assertTrue(result.exactGroundTruthMatch, "All ranked items must match ground truth exactly");
        assertTrue(result.top1MatchesGroundTruth, "Top-1 nearest neighbor must be the exact global minimum");

        // Verify distance strictly sorted non-decreasing
        for (int i = 0; i < result.routedHits.size() - 1; i++) {
            assertTrue(result.routedHits.get(i).distance <= result.routedHits.get(i + 1).distance + 1e-5f,
                    "Distances must be sorted monotonically");
        }
    }

    @Test
    @DisplayName("Case 2: Document chunks >= 10 (N=25) - Router transitions to HNSW with high recall")
    void testCase2OverThresholdChunks() {
        int chunkCount = 25; // Greater than 10
        List<DocItem> chunks = generateChunks(chunkCount, 123);
        List<Float> query = generateQuery(456);
        int k = 5;

        // Test with Manhattan Distance
        ThresholdComparisonResult result = comparisonService.evaluateDataset(
                chunks, query, k, "manhattan", 50
        );

        String report = comparisonService.formatReport(result);
        System.out.println(report);

        // Mathematical Assertions for Case 2 (N >= 10):
        assertFalse(result.isUnderThreshold, "Should identify that chunk count is >= 10");
        assertEquals("hnsw", result.routedAlgorithm, "Router must select HNSW when chunks >= 10");
        assertTrue(result.recallAtK >= 0.8f, "HNSW Recall@5 must be at least 80% (typically 100%)");
        assertTrue(result.distanceRatio >= 0.99 && result.distanceRatio <= 1.05,
                "Distance ratio should be extremely close to 1.0 (negligible approximation error)");
        assertTrue(result.top1MatchesGroundTruth, "Top-1 nearest neighbor should match ground truth");
    }

    @Test
    @DisplayName("Case Comparison: Multi-Metric Evaluation across Case 1 (N=8) and Case 2 (N=35)")
    void testMultiMetricCaseComparison() {
        String[] metrics = {"manhattan", "euclidean", "cosine"};

        for (String metric : metrics) {
            System.out.println("\n#################################################################################");
            System.out.println("          EVALUATING METRIC: " + metric.toUpperCase());
            System.out.println("#################################################################################");

            // Case 1: Less than 10 (N = 8)
            List<DocItem> smallChunks = generateChunks(8, 100);
            List<Float> query = generateQuery(200);
            ThresholdComparisonResult resSmall = comparisonService.evaluateDataset(smallChunks, query, 3, metric, 50);
            System.out.println(comparisonService.formatReport(resSmall));

            assertEquals("bruteforce", resSmall.routedAlgorithm);
            assertEquals(1.0f, resSmall.recallAtK, 1e-4f);
            assertEquals(1.0000, resSmall.distanceRatio, 1e-4);

            // Case 2: Greater than 10 (N = 35)
            List<DocItem> largeChunks = generateChunks(35, 300);
            ThresholdComparisonResult resLarge = comparisonService.evaluateDataset(largeChunks, query, 5, metric, 50);
            System.out.println(comparisonService.formatReport(resLarge));

            assertEquals("hnsw", resLarge.routedAlgorithm);
            assertTrue(resLarge.recallAtK >= 0.8f);
        }
    }

    @Test
    @DisplayName("Boundary Transition Test: Step-by-step from N=9 (Brute Force) to N=10 (HNSW)")
    void testBoundaryTransitionNineToTen() {
        List<Float> query = generateQuery(555);
        int k = 4;

        // At N = 9 chunks: System must use Brute Force
        List<DocItem> chunks9 = generateChunks(9, 789);
        ThresholdComparisonResult res9 = comparisonService.evaluateDataset(chunks9, query, k, "manhattan", 50);

        assertEquals(9, res9.chunkCount);
        assertTrue(res9.isUnderThreshold);
        assertEquals("bruteforce", res9.routedAlgorithm, "At N=9, algorithm must be Brute Force");
        assertEquals(1.0f, res9.recallAtK, 1e-4f, "At N=9, Recall must be 100%");

        // At N = 10 chunks: System must transition to HNSW
        List<DocItem> chunks10 = generateChunks(10, 789);
        ThresholdComparisonResult res10 = comparisonService.evaluateDataset(chunks10, query, k, "manhattan", 50);

        assertEquals(10, res10.chunkCount);
        assertFalse(res10.isUnderThreshold);
        assertEquals("hnsw", res10.routedAlgorithm, "At N=10, algorithm must transition to HNSW");
        assertTrue(res10.recallAtK >= 0.75f, "At N=10, HNSW Recall must be high");
        assertTrue(res10.distanceRatio <= 1.05, "Distance ratio should be near 1.0");

        System.out.printf("Boundary Transition Verified:%n  N=9  -> Algorithm: %s, Recall: %.1f%%%n  N=10 -> Algorithm: %s, Recall: %.1f%%%n",
                res9.routedAlgorithm, res9.recallAtK * 100.0f,
                res10.routedAlgorithm, res10.recallAtK * 100.0f);
    }

    @Test
    @DisplayName("Integration Test with Live DocumentDB: Dynamic threshold switching as chunks are added")
    void testLiveDocumentDbDynamicSwitching() {
        DocumentDB docDb = new DocumentDB();
        Random rng = new Random(888);

        // Step A: Insert 5 chunks (< 10)
        for (int i = 1; i <= 5; i++) {
            List<Float> emb = new ArrayList<>(DIMS);
            for (int d = 0; d < DIMS; d++) emb.add(rng.nextFloat());
            docDb.insert("Chunk #" + i, "Content of chunk " + i, emb, Distances::manhattan);
        }

        assertEquals(5, docDb.size());
        List<Float> query = generateQuery(999);

        // Search when count is 5
        DocumentDB.SearchResult searchRes5 = docDb.searchDetailed(query, 3, "manhattan");
        assertEquals("bruteforce", searchRes5.algoUsed(), "When size=5 (<10), search must use bruteforce");
        assertEquals(3, searchRes5.hits().size());

        // Step B: Insert 7 more chunks (total becomes 12 >= 10)
        for (int i = 6; i <= 12; i++) {
            List<Float> emb = new ArrayList<>(DIMS);
            for (int d = 0; d < DIMS; d++) emb.add(rng.nextFloat());
            docDb.insert("Chunk #" + i, "Content of chunk " + i, emb, Distances::manhattan);
        }

        assertEquals(12, docDb.size());

        // Search when count is 12
        DocumentDB.SearchResult searchRes12 = docDb.searchDetailed(query, 3, "manhattan");
        assertEquals("hnsw", searchRes12.algoUsed(), "When size=12 (>=10), search must switch to hnsw");
        assertEquals(3, searchRes12.hits().size());
    }
}
