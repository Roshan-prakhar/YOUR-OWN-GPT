package com.vectordb.algo;

import com.vectordb.model.VectorItem;

import java.util.*;

public class KDTree {

    private static class Node {
        VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    private Node insert(Node n, VectorItem v, int depth) {
        if (n == null) return new Node(v);
        int ax = depth % dims;
        if (v.emb.get(ax) < n.item.emb.get(ax))
            n.left  = insert(n.left,  v, depth + 1);
        else
            n.right = insert(n.right, v, depth + 1);
        return n;
    }

    public void insert(VectorItem v) {
        root = insert(root, v, 0);
    }

    private void knnSearch(Node n, List<Float> q, int k, int depth,
                           DistanceFn dist, PriorityQueue<DistHit> heap) {
        if (n == null) return;
        float dn = dist.apply(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek().dist()) {
            heap.add(new DistHit(dn, n.item.id));
            if (heap.size() > k) heap.poll();
        }
        int ax = depth % dims;
        float diff = q.get(ax) - n.item.emb.get(ax);
        Node closer  = diff < 0 ? n.left  : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        knnSearch(closer,  q, k, depth + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist())
            knnSearch(farther, q, k, depth + 1, dist, heap);
    }

    public List<DistHit> knn(List<Float> q, int k, DistanceFn dist) {
        PriorityQueue<DistHit> heap = new PriorityQueue<>(Comparator.reverseOrder());
        knnSearch(root, q, k, 0, dist, heap);
        List<DistHit> r = new ArrayList<>(heap);
        Collections.sort(r);
        return r;
    }

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) root = insert(root, v, 0);
    }
}
