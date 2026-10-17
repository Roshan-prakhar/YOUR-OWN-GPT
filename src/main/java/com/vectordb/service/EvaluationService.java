package com.vectordb.service;

import com.vectordb.algo.*;
import com.vectordb.db.VectorDB;
import com.vectordb.model.EvaluationResult;
import com.vectordb.model.VectorItem;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class EvaluationService {

    private static final float EPSILON = 1e-4f;

    /**
     * Evaluates HNSW retrieval against Brute Force ground truth on a set of items with the given metric.
     * Builds an HNSW index configured specifically for the specified metric.
     */
    public EvaluationResult evaluateDataset(List<VectorItem> dataset, List<Float> query, int k, String metric, int efSearch, int m, int efBuild) {
        DistanceFn distFn = Distances.get(metric);
        HNSW hnsw = new HNSW(m, efBuild);
        BruteForce bf = new BruteForce();

        Map<Integer, VectorItem> itemMap = new HashMap<>();
        for (VectorItem item : dataset) {
            itemMap.put(item.id, item);
            bf.insert(item);
            hnsw.insert(item, distFn);
        }

        // Measure Brute Force (Ground Truth)
        long t0 = System.nanoTime();
        List<DistHit> bfHits = bf.knn(query, k, distFn);
        long bfTimeUs = Math.max(1, (System.nanoTime() - t0) / 1000);

        // Measure HNSW Retrieval
        long t1 = System.nanoTime();
        List<DistHit> hnswHits = hnsw.knn(query, k, efSearch, distFn);
        long hnswTimeUs = Math.max(1, (System.nanoTime() - t1) / 1000);

        return buildResult(itemMap, query, distFn, metric, k, efSearch, dataset.size(), bfHits, hnswHits, bfTimeUs, hnswTimeUs);
    }

    /**
     * Evaluates the current VectorDB instance against Brute Force.
     */
    public EvaluationResult evaluateCurrentDb(VectorDB db, List<Float> query, int k, String metric, int efSearch) {
        DistanceFn distFn = Distances.get(metric);
        List<VectorItem> all = db.all();
        Map<Integer, VectorItem> itemMap = all.stream().collect(Collectors.toMap(v -> v.id, v -> v));

        BruteForce bf = new BruteForce();
        for (VectorItem vi : all) bf.insert(vi);

        long t0 = System.nanoTime();
        List<DistHit> bfHits = bf.knn(query, k, distFn);
        long bfTimeUs = Math.max(1, (System.nanoTime() - t0) / 1000);

        long t1 = System.nanoTime();
        HNSW hnsw = new HNSW(16, 200);
        for (VectorItem vi : all) hnsw.insert(vi, distFn);
        List<DistHit> hnswHits = hnsw.knn(query, k, efSearch, distFn);
        long hnswTimeUs = Math.max(1, (System.nanoTime() - t1) / 1000);

        return buildResult(itemMap, query, distFn, metric, k, efSearch, all.size(), bfHits, hnswHits, bfTimeUs, hnswTimeUs);
    }

    private EvaluationResult buildResult(
            Map<Integer, VectorItem> itemMap,
            List<Float> query,
            DistanceFn distFn,
            String metric,
            int k,
            int efSearch,
            int totalCount,
            List<DistHit> bfHits,
            List<DistHit> hnswHits,
            long bfTimeUs,
            long hnswTimeUs) {

        EvaluationResult res = new EvaluationResult();
        res.metric = metric;
        res.k = k;
        res.efSearch = efSearch;
        res.datasetSize = totalCount;
        res.bruteforceLatencyUs = bfTimeUs;
        res.hnswLatencyUs = hnswTimeUs;
        res.speedup = hnswTimeUs > 0 ? (double) bfTimeUs / hnswTimeUs : 1.0;

        // Ground Truth mapping: ID -> Rank
        Map<Integer, Integer> gtRanks = new HashMap<>();
        List<EvaluationResult.HitEvaluation> gtList = new ArrayList<>();
        boolean distanceValid = true;

        for (int i = 0; i < bfHits.size(); i++) {
            DistHit h = bfHits.get(i);
            int rank = i + 1;
            gtRanks.put(h.id(), rank);
            VectorItem item = itemMap.get(h.id());
            String meta = item != null ? item.metadata : "unknown";
            String cat = item != null ? item.category : "default";

            // Verify distance formula
            if (item != null) {
                float expected = distFn.apply(query, item.emb);
                if (Math.abs(expected - h.dist()) > EPSILON) distanceValid = false;
            }

            gtList.add(new EvaluationResult.HitEvaluation(rank, h.id(), meta, cat, h.dist(), true, rank));
        }
        res.groundTruth = gtList;

        // HNSW Retrieval mapping
        List<EvaluationResult.HitEvaluation> hnswList = new ArrayList<>();
        int intersectionCount = 0;

        for (int i = 0; i < hnswHits.size(); i++) {
            DistHit h = hnswHits.get(i);
            int rank = i + 1;
            boolean inGt = gtRanks.containsKey(h.id());
            int gtRank = gtRanks.getOrDefault(h.id(), -1);
            if (inGt) intersectionCount++;

            VectorItem item = itemMap.get(h.id());
            String meta = item != null ? item.metadata : "unknown";
            String cat = item != null ? item.category : "default";

            if (item != null) {
                float expected = distFn.apply(query, item.emb);
                if (Math.abs(expected - h.dist()) > EPSILON) distanceValid = false;
            }

            hnswList.add(new EvaluationResult.HitEvaluation(rank, h.id(), meta, cat, h.dist(), inGt, gtRank));
        }
        res.hnswRetrieval = hnswList;

        int effectiveK = Math.min(k, totalCount);
        res.recallAtK = effectiveK > 0 ? (float) intersectionCount / effectiveK : 1.0f;
        res.precisionAtK = hnswHits.size() > 0 ? (float) intersectionCount / hnswHits.size() : 1.0f;
        res.top1Match = !bfHits.isEmpty() && !hnswHits.isEmpty() && bfHits.get(0).id() == hnswHits.get(0).id();
        res.distanceCalculationsValid = distanceValid;

        return res;
    }

    /**
     * Formats the evaluation results into an ASCII table report.
     */
    public String formatReport(EvaluationResult res) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n=========================================================================================\n");
        sb.append(String.format("          HNSW vs BRUTE FORCE RETRIEVAL EVALUATION [%s]\n", res.metric.toUpperCase()));
        sb.append("=========================================================================================\n");
        sb.append(String.format("Dataset Size : %d items | K Target: %d | efSearch: %d\n", res.datasetSize, res.k, res.efSearch));
        sb.append(String.format("Recall@%d     : %.1f%%  | Precision@%d: %.1f%%\n", res.k, res.recallAtK * 100.0f, res.k, res.precisionAtK * 100.0f));
        sb.append(String.format("Top-1 Match  : %s     | Distance Check: %s\n", res.top1Match ? "YES (Exact match)" : "NO", res.distanceCalculationsValid ? "PASSED (Exact)" : "FAILED"));
        sb.append(String.format("Latencies    : BruteForce = %d µs | HNSW = %d µs (Speedup: %.2fx)\n", res.bruteforceLatencyUs, res.hnswLatencyUs, res.speedup));
        sb.append("-----------------------------------------------------------------------------------------\n");
        sb.append("HNSW RETRIEVED ITEMS (vs Ground Truth):\n");
        sb.append(String.format("%-5s | %-4s | %-8s | %-12s | %-8s | %-40s\n",
                "Rank", "ID", "Category", "Distance", "BF Rank", "Metadata Snippet"));
        sb.append("------+------+----------+--------------+---------+--------------------------------------\n");

        for (EvaluationResult.HitEvaluation h : res.hnswRetrieval) {
            String snippet = h.metadata.length() > 38 ? h.metadata.substring(0, 35) + "..." : h.metadata;
            String bfRankStr = h.inGroundTruth ? "#" + h.groundTruthRank : "MISS";
            sb.append(String.format("%-5d | %-4d | %-8s | %-12.5f | %-8s | %-40s\n",
                    h.rank, h.id, h.category, h.distance, bfRankStr, snippet));
        }

        sb.append("=========================================================================================\n");
        return sb.toString();
    }
}
