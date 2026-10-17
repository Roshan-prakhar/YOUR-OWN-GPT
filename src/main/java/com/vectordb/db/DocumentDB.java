package com.vectordb.db;

import com.vectordb.algo.*;
import com.vectordb.model.DocItem;
import com.vectordb.model.VectorItem;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

@Component
public class DocumentDB {

    public record DocHit(float dist, DocItem doc) {}
    public record SearchResult(List<DocHit> hits, String algoUsed, int totalChunks, long latencyUs) {}

    public static final int THRESHOLD = 10;

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW       hnsw = new HNSW(16, 200);
    private final BruteForce bf   = new BruteForce();
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    private int dims   = 0;

    public int insert(String title, String text, List<Float> emb) {
        return insert(title, text, emb, Distances::cosine);
    }

    public int insert(String title, String text, List<Float> emb, DistanceFn dist) {
        lock.lock();
        try {
            if (dims == 0) dims = emb.size();
            DocItem item = new DocItem(nextId++, title, text, new ArrayList<>(emb));
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", item.emb);
            hnsw.insert(vi, dist);
            bf.insert(vi);
            return item.id;
        } finally { lock.unlock(); }
    }

    public List<DocHit> search(List<Float> q, int k) {
        return search(q, k, "cosine");
    }

    public List<DocHit> search(List<Float> q, int k, String metric) {
        return searchDetailed(q, k, metric).hits();
    }

    public SearchResult searchDetailed(List<Float> q, int k, String metric) {
        lock.lock();
        try {
            if (store.isEmpty()) return new SearchResult(Collections.emptyList(), "none", 0, 0);
            DistanceFn dfn = Distances.get(metric);
            String algo = (store.size() < THRESHOLD) ? "bruteforce" : "hnsw";

            long t0 = System.nanoTime();
            List<DistHit> raw = (store.size() < THRESHOLD)
                    ? bf.knn(q, k, dfn)
                    : hnsw.knn(q, k, 50, dfn);
            long latencyUs = Math.max(1, (System.nanoTime() - t0) / 1000);

            List<DocHit> out = new ArrayList<>();
            for (DistHit h : raw) {
                DocItem doc = store.get(h.id());
                if (doc != null) out.add(new DocHit(h.dist(), doc));
            }
            return new SearchResult(out, algo, store.size(), latencyUs);
        } finally { lock.unlock(); }
    }

    public HNSW getHnsw() { return hnsw; }
    public BruteForce getBf() { return bf; }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally { lock.unlock(); }
    }

    public List<DocItem> all() {
        lock.lock();
        try { return new ArrayList<>(store.values()); }
        finally { lock.unlock(); }
    }

    public int size() {
        lock.lock();
        try { return store.size(); }
        finally { lock.unlock(); }
    }

    public int getDims() { return dims; }
}
