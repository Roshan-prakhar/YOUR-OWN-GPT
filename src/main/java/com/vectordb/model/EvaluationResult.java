package com.vectordb.model;

import java.util.List;

public class EvaluationResult {

    public String metric;
    public int k;
    public int efSearch;
    public int datasetSize;
    public float recallAtK;
    public float precisionAtK;
    public boolean top1Match;
    public long bruteforceLatencyUs;
    public long hnswLatencyUs;
    public double speedup;
    public boolean distanceCalculationsValid;
    public List<HitEvaluation> groundTruth;
    public List<HitEvaluation> hnswRetrieval;

    public static class HitEvaluation {
        public int rank;
        public int id;
        public String metadata;
        public String category;
        public float distance;
        public boolean inGroundTruth;
        public int groundTruthRank;

        public HitEvaluation() {}

        public HitEvaluation(int rank, int id, String metadata, String category, float distance, boolean inGroundTruth, int groundTruthRank) {
            this.rank = rank;
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.distance = distance;
            this.inGroundTruth = inGroundTruth;
            this.groundTruthRank = groundTruthRank;
        }
    }
}
