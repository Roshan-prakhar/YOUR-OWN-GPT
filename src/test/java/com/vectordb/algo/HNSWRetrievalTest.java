package com.vectordb.algo;

import com.vectordb.model.VectorItem;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("HNSW Retrieval Calculation & Brute Force Comparison Tests")
public class HNSWRetrievalTest {

    private static final float EPSILON = 1e-4f;

    @Test
    @DisplayName("HNSW Retrieval with Manhattan Distance: Verification against Brute Force ground truth")
    void testHnswManhattanRetrievalAgainstBruteForce() {
        // Instantiate HNSW and BruteForce
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        DistanceFn manhattan = Distances::manhattan;

        // Populate with synthetic 8-dimensional test dataset
        Random rng = new Random(42);
        int numVectors = 60;
        int dims = 8;

        for (int i = 1; i <= numVectors; i++) {
            List<Float> emb = new ArrayList<>(dims);
            for (int d = 0; d < dims; d++) {
                emb.add(rng.nextFloat() * 10.0f - 5.0f); // values between -5 and 5
            }
            VectorItem item = new VectorItem(i, "item-" + i, "category-" + (i % 3), emb);
            hnsw.insert(item, manhattan);
            bf.insert(item);
        }

        // Generate query vector
        List<Float> query = new ArrayList<>(dims);
        for (int d = 0; d < dims; d++) {
            query.add(rng.nextFloat() * 10.0f - 5.0f);
        }

        int k = 5;
        int efSearch = 50;

        // Ground Truth via Brute Force
        List<DistHit> groundTruth = bf.knn(query, k, manhattan);
        // HNSW Approximate Nearest Neighbors
        List<DistHit> hnswHits = hnsw.knn(query, k, efSearch, manhattan);

        assertEquals(k, groundTruth.size(), "Ground truth must return k items");
        assertEquals(k, hnswHits.size(), "HNSW must return k items");

        Set<Integer> gtIds = groundTruth.stream().map(DistHit::id).collect(Collectors.toSet());
        Set<Integer> hnswIds = hnswHits.stream().map(DistHit::id).collect(Collectors.toSet());

        // Calculate intersection
        Set<Integer> intersection = new HashSet<>(gtIds);
        intersection.retainAll(hnswIds);

        float recallAtK = (float) intersection.size() / k;
        System.out.println("HNSW Manhattan Retrieval Recall@" + k + ": " + (recallAtK * 100.0f) + "%");

        // Verify distance calculations: every hit's distance must equal manual Manhattan distance
        for (DistHit hit : hnswHits) {
            VectorItem item = bf.getItems().stream().filter(v -> v.id == hit.id()).findFirst().orElseThrow();
            float expectedDist = Distances.manhattan(query, item.emb);
            assertEquals(expectedDist, hit.dist(), EPSILON, "HNSW reported distance must match exact Manhattan formula");
        }

        // HNSW with ef=50 should have >= 80% recall on this dataset
        assertTrue(recallAtK >= 0.8f, "Recall@5 with efSearch=50 should be at least 80%");
    }

    @Test
    @DisplayName("HNSW Recall@K improvement as efSearch increases under Manhattan distance")
    void testEfSearchRecallScalingManhattan() {
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        DistanceFn manhattan = Distances::manhattan;

        Random rng = new Random(100);
        int numVectors = 100;
        int dims = 16;

        for (int i = 1; i <= numVectors; i++) {
            List<Float> emb = new ArrayList<>(dims);
            for (int d = 0; d < dims; d++) {
                emb.add(rng.nextFloat());
            }
            VectorItem item = new VectorItem(i, "v" + i, "group", emb);
            hnsw.insert(item, manhattan);
            bf.insert(item);
        }

        List<Float> query = new ArrayList<>(dims);
        for (int d = 0; d < dims; d++) {
            query.add(rng.nextFloat());
        }

        int k = 10;
        List<DistHit> groundTruth = bf.knn(query, k, manhattan);
        Set<Integer> gtIds = groundTruth.stream().map(DistHit::id).collect(Collectors.toSet());

        // Test with efSearch = 5, 20, 60
        int[] efValues = {5, 20, 60};
        float prevRecall = 0.0f;

        for (int ef : efValues) {
            List<DistHit> hits = hnsw.knn(query, k, ef, manhattan);
            Set<Integer> hitsIds = hits.stream().map(DistHit::id).collect(Collectors.toSet());
            hitsIds.retainAll(gtIds);
            float recall = (float) hitsIds.size() / k;
            System.out.printf("efSearch=%d -> Recall@%d = %.1f%%%n", ef, k, recall * 100.0f);
            assertTrue(recall >= prevRecall - 0.1f, "Recall should generally improve or remain steady with higher ef");
            prevRecall = recall;
        }

        // High ef search should give very high recall
        List<DistHit> highEfHits = hnsw.knn(query, k, 100, manhattan);
        Set<Integer> highEfIds = highEfHits.stream().map(DistHit::id).collect(Collectors.toSet());
        highEfIds.retainAll(gtIds);
        float highRecall = (float) highEfIds.size() / k;
        assertTrue(highRecall >= 0.9f, "High efSearch should achieve >= 90% Recall@10");
    }

    @Test
    @DisplayName("HNSW Multi-Metric Retrieval Comparison: Manhattan vs Euclidean vs Cosine")
    void testMultiMetricRetrievalComparison() {
        String[] metrics = {"manhattan", "euclidean", "cosine"};
        Random rng = new Random(2024);
        int dims = 8;
        int n = 50;

        List<VectorItem> dataset = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            List<Float> emb = new ArrayList<>(dims);
            for (int d = 0; d < dims; d++) emb.add(rng.nextFloat() + 0.1f);
            dataset.add(new VectorItem(i, "doc-" + i, "data", emb));
        }

        List<Float> query = new ArrayList<>(dims);
        for (int d = 0; d < dims; d++) query.add(rng.nextFloat() + 0.1f);

        int k = 5;

        for (String metricName : metrics) {
            DistanceFn fn = Distances.get(metricName);
            HNSW hnsw = new HNSW(16, 200);
            BruteForce bf = new BruteForce();

            for (VectorItem vi : dataset) {
                hnsw.insert(vi, fn);
                bf.insert(vi);
            }

            long t0 = System.nanoTime();
            List<DistHit> gt = bf.knn(query, k, fn);
            long bfTimeUs = (System.nanoTime() - t0) / 1000;

            t0 = System.nanoTime();
            List<DistHit> hnswHits = hnsw.knn(query, k, 50, fn);
            long hnswTimeUs = (System.nanoTime() - t0) / 1000;

            Set<Integer> gtSet = gt.stream().map(DistHit::id).collect(Collectors.toSet());
            Set<Integer> hnswSet = hnswHits.stream().map(DistHit::id).collect(Collectors.toSet());
            hnswSet.retainAll(gtSet);

            float recall = (float) hnswSet.size() / k;
            System.out.printf("[%s] Recall@%d: %.1f%% | BF Latency: %d µs | HNSW Latency: %d µs%n",
                    metricName.toUpperCase(), k, recall * 100.0f, bfTimeUs, hnswTimeUs);

            assertTrue(recall >= 0.8f, "Recall for " + metricName + " should be at least 80%");
        }
    }
}
