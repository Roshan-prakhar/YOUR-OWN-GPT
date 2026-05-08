package com.vectordb.algo;

import com.vectordb.model.VectorItem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BruteForce {

    private final List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) {
        items.add(v);
    }

    public List<DistHit> knn(List<Float> q, int k, DistanceFn dist) {
        List<DistHit> r = new ArrayList<>(items.size());
        for (VectorItem v : items)
            r.add(new DistHit(dist.apply(q, v.emb), v.id));
        Collections.sort(r);
        return r.size() > k ? new ArrayList<>(r.subList(0, k)) : r;
    }

    public void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    public List<VectorItem> getItems() {
        return Collections.unmodifiableList(items);
    }
}
