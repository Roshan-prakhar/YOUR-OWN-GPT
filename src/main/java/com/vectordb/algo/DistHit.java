package com.vectordb.algo;

public record DistHit(float dist, int id) implements Comparable<DistHit> {
    @Override
    public int compareTo(DistHit o) {
        return Float.compare(this.dist, o.dist);
    }
}
