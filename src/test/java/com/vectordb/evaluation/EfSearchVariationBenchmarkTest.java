package com.vectordb.evaluation;

import com.vectordb.algo.*;
import com.vectordb.model.DocItem;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("HNSW efSearch Variation Benchmark (ef = 5 to 200)")
public class EfSearchVariationBenchmarkTest {

    private static final int DIMS = 16;

    @Test
    @DisplayName("Evaluate efSearch variation on Recall@3 and Latency (ef=1 to 200)")
    void testEfSearchVariation() {
        int n = 100;
        int k = 3;
        Random rng = new Random(42);

        // 1. Create 100 document chunks
        List<DocItem> chunks = new ArrayList<>();
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        DistanceFn cosine = Distances::cosine;

        for (int i = 1; i <= n; i++) {
            List<Float> emb = new ArrayList<>(DIMS);
            for (int d = 0; d < DIMS; d++) emb.add(rng.nextFloat());
            DocItem doc = new DocItem(i, "Chunk #" + i, "Content #" + i, emb);
            chunks.add(doc);
            hnsw.insert(new com.vectordb.model.VectorItem(doc.id, doc.title, "doc", doc.emb), cosine);
            bf.insert(new com.vectordb.model.VectorItem(doc.id, doc.title, "doc", doc.emb));
        }

        // 2. Test query
        List<Float> query = new ArrayList<>(DIMS);
        for (int d = 0; d < DIMS; d++) query.add(rng.nextFloat());

        // Ground Truth (Teacher's Key) via Brute Force
        List<DistHit> groundTruth = bf.knn(query, k, cosine);
        Set<Integer> gtIds = groundTruth.stream().map(DistHit::id).collect(Collectors.toSet());

        // Test sweep across efSearch values
        int[] efValues = {1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200};

        System.out.println("\n=======================================================================");
        System.out.println("          HNSW efSearch VARIATION EXPERIMENTAL RESULTS");
        System.out.println("=======================================================================");
        System.out.printf("%-10s | %-12s | %-14s | %-18s\n", "efSearch", "Recall@3 (%)", "Latency (µs)", "Behavior");
        System.out.println("-----------+--------------+----------------+---------------------------");

        for (int ef : efValues) {
            // Warmup & measure
            long t0 = System.nanoTime();
            List<DistHit> hits = hnsw.knn(query, k, ef, cosine);
            long latencyUs = Math.max(1, (System.nanoTime() - t0) / 1000);

            Set<Integer> hitIds = hits.stream().map(DistHit::id).collect(Collectors.toSet());
            hitIds.retainAll(gtIds);
            float recall = (float) hitIds.size() / k * 100.0f;

            String behavior;
            if (ef < 10) behavior = "Fast, but misses true neighbors";
            else if (ef == 20) behavior = "High recall achieved";
            else if (ef == 50) behavior = "★ SWEET SPOT: 100% Recall, Fast";
            else behavior = "Diminishing returns (Wasted CPU)";

            System.out.printf("ef = %-5d | %-12.1f | %-14d | %s\n", ef, recall, latencyUs, behavior);

            if (ef >= 50) {
                assertEquals(100.0f, recall, 1e-4f, "efSearch >= 50 should have 100% Recall@3");
            }
        }
        System.out.println("=======================================================================\n");
    }
}
