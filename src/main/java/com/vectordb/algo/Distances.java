package com.vectordb.algo;

import java.util.List;

public class Distances {

    public static float euclidean(List<Float> a, List<Float> b) {
        float s = 0;
        for (int i = 0; i < a.size(); i++) {
            float d = a.get(i) - b.get(i);
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    public static float cosine(List<Float> a, List<Float> b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.size(); i++) {
            float ai = a.get(i), bi = b.get(i);
            dot += ai * bi;
            na  += ai * ai;
            nb  += bi * bi;
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
    }

    public static float manhattan(List<Float> a, List<Float> b) {
        float s = 0;
        for (int i = 0; i < a.size(); i++)
            s += Math.abs(a.get(i) - b.get(i));
        return s;
    }

    public static DistanceFn get(String metric) {
        return switch (metric) {
            case "cosine"    -> Distances::cosine;
            case "manhattan" -> Distances::manhattan;
            default          -> Distances::euclidean;
        };
    }
}
