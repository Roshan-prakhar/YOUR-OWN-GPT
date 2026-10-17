package com.vectordb.service;

import com.vectordb.algo.*;
import com.vectordb.db.DocumentDB;
import com.vectordb.model.DocItem;
import com.vectordb.model.ThresholdComparisonResult;
import com.vectordb.model.VectorItem;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class ThresholdComparisonService {

    private static final float EPSILON = 1e-4f;

    /**
     * Evaluates a collection of document chunks under the threshold routing policy (threshold = 10).
     */
    public ThresholdComparisonResult evaluateDataset(
            List<DocItem> chunks,
            List<Float> query,
            int k,
            String metric,
            int efSearch) {

        DistanceFn distFn = Distances.get(metric);
        int totalChunks = chunks.size();
        int threshold = DocumentDB.THRESHOLD;
        boolean isUnderThreshold = totalChunks < threshold;
        String routedAlgo = isUnderThreshold ? "bruteforce" : "hnsw";

        // Build both indexes with the specified metric
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        Map<Integer, DocItem> itemMap = new HashMap<>();

        for (DocItem item : chunks) {
            itemMap.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, item.title, "doc", item.emb);
            hnsw.insert(vi, distFn);
            bf.insert(vi);
        }

        // 1. Ground Truth via Brute Force
        long t0 = System.nanoTime();
        List<DistHit> bfRaw = bf.knn(query, k, distFn);
        long bfTimeUs = Math.max(1, (System.nanoTime() - t0) / 1000);

        // 2. HNSW Search
        long t1 = System.nanoTime();
        List<DistHit> hnswRaw = hnsw.knn(query, k, efSearch, distFn);
        long hnswTimeUs = Math.max(1, (System.nanoTime() - t1) / 1000);

        // 3. Routed Execution
        List<DistHit> routedRaw = isUnderThreshold ? bfRaw : hnswRaw;
        long routedTimeUs = isUnderThreshold ? bfTimeUs : hnswTimeUs;

        ThresholdComparisonResult res = new ThresholdComparisonResult();
        res.chunkCount = totalChunks;
        res.threshold = threshold;
        res.isUnderThreshold = isUnderThreshold;
        res.routedAlgorithm = routedAlgo;
        res.metric = metric;
        res.k = k;
        res.routedLatencyUs = routedTimeUs;
        res.bruteforceLatencyUs = bfTimeUs;
        res.hnswLatencyUs = hnswTimeUs;

        // Ground Truth Map
        Map<Integer, Integer> gtRankMap = new HashMap<>();
        List<ThresholdComparisonResult.DocHitDetail> gtDetails = new ArrayList<>();
        double gtDistSum = 0.0;

        for (int i = 0; i < bfRaw.size(); i++) {
            DistHit h = bfRaw.get(i);
            int rank = i + 1;
            gtRankMap.put(h.id(), rank);
            DocItem doc = itemMap.get(h.id());
            String title = doc != null ? doc.title : "Doc #" + h.id();
            gtDistSum += h.dist();
            gtDetails.add(new ThresholdComparisonResult.DocHitDetail(rank, h.id(), title, h.dist(), true, rank));
        }
        res.groundTruthHits = gtDetails;

        // Routed Details
        List<ThresholdComparisonResult.DocHitDetail> routedDetails = new ArrayList<>();
        int intersection = 0;
        double routedDistSum = 0.0;
        boolean allRanksMatch = (routedRaw.size() == bfRaw.size());

        for (int i = 0; i < routedRaw.size(); i++) {
            DistHit h = routedRaw.get(i);
            int rank = i + 1;
            boolean inGt = gtRankMap.containsKey(h.id());
            int gtRank = gtRankMap.getOrDefault(h.id(), -1);
            if (inGt) intersection++;
            if (gtRank != rank) allRanksMatch = false;

            DocItem doc = itemMap.get(h.id());
            String title = doc != null ? doc.title : "Doc #" + h.id();
            routedDistSum += h.dist();
            routedDetails.add(new ThresholdComparisonResult.DocHitDetail(rank, h.id(), title, h.dist(), inGt, gtRank));
        }
        res.routedHits = routedDetails;

        // Alternative Details (what the other algorithm produced)
        List<DistHit> altRaw = isUnderThreshold ? hnswRaw : bfRaw;
        List<ThresholdComparisonResult.DocHitDetail> altDetails = new ArrayList<>();
        for (int i = 0; i < altRaw.size(); i++) {
            DistHit h = altRaw.get(i);
            int rank = i + 1;
            boolean inGt = gtRankMap.containsKey(h.id());
            int gtRank = gtRankMap.getOrDefault(h.id(), -1);
            DocItem doc = itemMap.get(h.id());
            String title = doc != null ? doc.title : "Doc #" + h.id();
            altDetails.add(new ThresholdComparisonResult.DocHitDetail(rank, h.id(), title, h.dist(), inGt, gtRank));
        }
        res.alternativeHits = altDetails;

        int effectiveK = Math.min(k, totalChunks);
        res.recallAtK = effectiveK > 0 ? (float) intersection / effectiveK : 1.0f;
        res.precisionAtK = routedRaw.size() > 0 ? (float) intersection / routedRaw.size() : 1.0f;
        res.distanceRatio = gtDistSum > 1e-9 ? routedDistSum / gtDistSum : 1.0;
        res.exactGroundTruthMatch = allRanksMatch && (intersection == effectiveK);
        res.top1MatchesGroundTruth = !bfRaw.isEmpty() && !routedRaw.isEmpty() && (bfRaw.get(0).id() == routedRaw.get(0).id());

        return res;
    }

    /**
     * Formats the comparison result into a detailed ASCII table.
     */
    public String formatReport(ThresholdComparisonResult res) {
        StringBuilder sb = new StringBuilder();
        String regime = res.isUnderThreshold
                ? String.format("CASE 1: DOCUMENT CHUNKS < 10 (%d Chunks)", res.chunkCount)
                : String.format("CASE 2: DOCUMENT CHUNKS >= 10 (%d Chunks)", res.chunkCount);

        sb.append("\n=========================================================================================\n");
        sb.append("   " + regime + " — " + res.metric.toUpperCase() + "\n");
        sb.append("=========================================================================================\n");
        sb.append(String.format("Router Policy  : Index size %d %s threshold %d -> Selected: [%s]\n",
                res.chunkCount, res.isUnderThreshold ? "<" : ">=", res.threshold, res.routedAlgorithm.toUpperCase()));
        sb.append(String.format("Recall@%d       : %.1f%%  | Precision@%d: %.1f%%\n",
                res.k, res.recallAtK * 100.0f, res.k, res.precisionAtK * 100.0f));
        sb.append(String.format("Distance Ratio : %.4f (1.0000 = exact ground truth)\n", res.distanceRatio));
        sb.append(String.format("Top-1 Match    : %s | Exact Ranking Match: %s\n",
                res.top1MatchesGroundTruth ? "YES" : "NO", res.exactGroundTruthMatch ? "YES" : "NO"));
        sb.append(String.format("Latencies      : Routed (%s) = %d µs | BF = %d µs | HNSW = %d µs\n",
                res.routedAlgorithm, res.routedLatencyUs, res.bruteforceLatencyUs, res.hnswLatencyUs));
        sb.append("-----------------------------------------------------------------------------------------\n");
        sb.append(String.format("%-5s | %-4s | %-14s | %-9s | %-35s\n",
                "Rank", "ID", "Distance (" + res.metric + ")", "BF Rank", "Chunk Title"));
        sb.append("------+------+----------------+-----------+----------------------------------------------\n");

        for (ThresholdComparisonResult.DocHitDetail h : res.routedHits) {
            String title = h.title.length() > 35 ? h.title.substring(0, 32) + "..." : h.title;
            String bfRank = h.inGroundTruth ? "#" + h.groundTruthRank : "MISS";
            sb.append(String.format("%-5d | %-4d | %-14.5f | %-9s | %-35s\n",
                    h.rank, h.id, h.distance, bfRank, title));
        }

        sb.append("=========================================================================================\n");
        return sb.toString();
    }
}
