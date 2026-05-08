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

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW       hnsw = new HNSW(16, 200);
    private final BruteForce bf   = new BruteForce();
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    private int dims   = 0;

    public int insert(String title, String text, List<Float> emb) {
        lock.lock();
        try {
            if (dims == 0) dims = emb.size();
            DocItem item = new DocItem(nextId++, title, text, new ArrayList<>(emb));
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", item.emb);
            hnsw.insert(vi, Distances::cosine);
            bf.insert(vi);
            return item.id;
        } finally { lock.unlock(); }
    }

    public List<DocHit> search(List<Float> q, int k) {
        lock.lock();
        try {
            if (store.isEmpty()) return Collections.emptyList();
            List<DistHit> raw = (store.size() < 10)
                    ? bf.knn(q, k, Distances::cosine)
                    : hnsw.knn(q, k, 50, Distances::cosine);
            List<DocHit> out = new ArrayList<>();
            for (DistHit h : raw) {
                DocItem doc = store.get(h.id());
                if (doc != null) out.add(new DocHit(h.dist(), doc));
            }
            return out;
        } finally { lock.unlock(); }
    }

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
