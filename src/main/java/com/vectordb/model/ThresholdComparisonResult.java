package com.vectordb.model;

import java.util.List;

public class ThresholdComparisonResult {

    public int chunkCount;
    public int threshold;
    public boolean isUnderThreshold;
    public String routedAlgorithm;
    public String metric;
    public int k;

    // Mathematical Evaluation Metrics
    public float recallAtK;
    public float precisionAtK;
    public double distanceRatio; // sum(d_routed) / sum(d_groundTruth)
    public boolean exactGroundTruthMatch;
    public boolean top1MatchesGroundTruth;

    // Latency Comparison
    public long routedLatencyUs;
    public long bruteforceLatencyUs;
    public long hnswLatencyUs;

    // Hit Details
    public List<DocHitDetail> routedHits;
    public List<DocHitDetail> groundTruthHits;
    public List<DocHitDetail> alternativeHits; // What the other algorithm produced

    public static class DocHitDetail {
        public int rank;
        public int id;
        public String title;
        public float distance;
        public boolean inGroundTruth;
        public int groundTruthRank;

        public DocHitDetail() {}

        public DocHitDetail(int rank, int id, String title, float distance, boolean inGroundTruth, int groundTruthRank) {
            this.rank = rank;
            this.id = id;
            this.title = title;
            this.distance = distance;
            this.inGroundTruth = inGroundTruth;
            this.groundTruthRank = groundTruthRank;
        }
    }
}
