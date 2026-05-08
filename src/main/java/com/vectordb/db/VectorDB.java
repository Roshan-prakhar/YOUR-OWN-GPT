package com.vectordb.db;

import com.vectordb.algo.*;
import com.vectordb.model.VectorItem;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

@Component
public class VectorDB {

    public static final int DIMS = 16;

    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForce bf   = new BruteForce();
    private final KDTree     kdt  = new KDTree(DIMS);
    private final HNSW       hnsw = new HNSW(16, 200);
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;

    public int insert(String meta, String cat, List<Float> emb, DistanceFn dist) {
        lock.lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, cat, new ArrayList<>(emb));
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally { lock.unlock(); }
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally { lock.unlock(); }
    }

    public static class Hit {
        public int id;
        public String meta, cat;
        public List<Float> emb;
        public float dist;

        public Hit(int id, String meta, String cat, List<Float> emb, float dist) {
            this.id = id; this.meta = meta; this.cat = cat;
            this.emb = emb; this.dist = dist;
        }
    }

    public static class SearchOut {
        public List<Hit> hits;
        public long latencyUs;
        public String algo, metric;

        public SearchOut(List<Hit> hits, long latencyUs, String algo, String metric) {
            this.hits = hits; this.latencyUs = latencyUs;
            this.algo = algo; this.metric = metric;
        }
    }

    public SearchOut search(List<Float> q, int k, String metric, String algo) {
        lock.lock();
        try {
            DistanceFn dfn = Distances.get(metric);
            long t0 = System.nanoTime();

            List<DistHit> raw = switch (algo) {
                case "bruteforce" -> bf.knn(q, k, dfn);
                case "kdtree"     -> kdt.knn(q, k, dfn);
                default           -> hnsw.knn(q, k, 50, dfn);
            };

            long us = (System.nanoTime() - t0) / 1000;
            List<Hit> hits = new ArrayList<>();
            for (DistHit h : raw) {
                VectorItem v = store.get(h.id());
                if (v != null)
                    hits.add(new Hit(v.id, v.metadata, v.category, v.emb, h.dist()));
            }
            return new SearchOut(hits, us, algo, metric);
        } finally { lock.unlock(); }
    }

    public static class BenchOut {
        public long bfUs, kdUs, hnswUs;
        public int itemCount;

        public BenchOut(long bfUs, long kdUs, long hnswUs, int itemCount) {
            this.bfUs = bfUs; this.kdUs = kdUs;
            this.hnswUs = hnswUs; this.itemCount = itemCount;
        }
    }

    public BenchOut benchmark(List<Float> q, int k, String metric) {
        lock.lock();
        try {
            DistanceFn dfn = Distances.get(metric);
            long t;

            t = System.nanoTime(); bf.knn(q, k, dfn);
            long bfUs = (System.nanoTime() - t) / 1000;

            t = System.nanoTime(); kdt.knn(q, k, dfn);
            long kdUs = (System.nanoTime() - t) / 1000;

            t = System.nanoTime(); hnsw.knn(q, k, 50, dfn);
            long hnswUs = (System.nanoTime() - t) / 1000;

            return new BenchOut(bfUs, kdUs, hnswUs, store.size());
        } finally { lock.unlock(); }
    }

    public List<VectorItem> all() {
        lock.lock();
        try { return new ArrayList<>(store.values()); }
        finally { lock.unlock(); }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.lock();
        try { return hnsw.getInfo(); }
        finally { lock.unlock(); }
    }

    public int size() {
        lock.lock();
        try { return store.size(); }
        finally { lock.unlock(); }
    }
}
