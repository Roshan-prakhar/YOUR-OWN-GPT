package com.vectordb.algo;

import com.vectordb.model.VectorItem;

import java.util.*;

public class HNSW {

    private static class Node {
        VectorItem item;
        int maxLyr;
        List<List<Integer>> nbrs;

        Node(VectorItem item, int maxLyr) {
            this.item   = item;
            this.maxLyr = maxLyr;
            this.nbrs   = new ArrayList<>();
            for (int i = 0; i <= maxLyr; i++)
                this.nbrs.add(new ArrayList<>());
        }
    }

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int    M, M0, efBuild;
    private final float  mL;
    private int topLayer = -1;
    private int entryPt  = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.M       = m;
        this.M0      = 2 * m;
        this.efBuild = efBuild;
        this.mL      = 1.0f / (float) Math.log(m);
    }

    private int randLevel() {
        float u = Math.max(rng.nextFloat(), 1e-10f);
        return (int) Math.floor(-Math.log(u) * mL);
    }

    private List<DistHit> searchLayer(List<Float> q, int ep, int ef, int lyr, DistanceFn dist) {
        Set<Integer> vis   = new HashSet<>();
        PriorityQueue<DistHit> cands = new PriorityQueue<>();
        PriorityQueue<DistHit> found = new PriorityQueue<>(Comparator.reverseOrder());

        float d0 = dist.apply(q, graph.get(ep).item.emb);
        vis.add(ep);
        cands.add(new DistHit(d0, ep));
        found.add(new DistHit(d0, ep));

        while (!cands.isEmpty()) {
            DistHit c = cands.poll();
            if (found.size() >= ef && c.dist() > found.peek().dist()) break;

            Node cNode = graph.get(c.id());
            if (cNode == null || lyr >= cNode.nbrs.size()) continue;

            for (int nid : cNode.nbrs.get(lyr)) {
                if (vis.contains(nid) || !graph.containsKey(nid)) continue;
                vis.add(nid);
                float nd = dist.apply(q, graph.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek().dist()) {
                    cands.add(new DistHit(nd, nid));
                    found.add(new DistHit(nd, nid));
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<DistHit> res = new ArrayList<>(found);
        Collections.sort(res);
        return res;
    }

    private List<Integer> selectNbrs(List<DistHit> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        for (int i = 0; i < Math.min(cands.size(), maxM); i++)
            r.add(cands.get(i).id());
        return r;
    }

    public void insert(VectorItem item, DistanceFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        graph.put(id, new Node(item, lvl));

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<DistHit> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id();
            }
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<DistHit> W  = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM         = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);
            graph.get(id).nbrs.set(lc, new ArrayList<>(sel));

            for (int nid : sel) {
                Node nNode = graph.get(nid);
                if (nNode == null) continue;
                while (nNode.nbrs.size() <= lc) nNode.nbrs.add(new ArrayList<>());
                List<Integer> conn = nNode.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<DistHit> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cn = graph.get(c);
                        if (cn != null)
                            ds.add(new DistHit(dist.apply(nNode.item.emb, cn.item.emb), c));
                    }
                    Collections.sort(ds);
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++)
                        conn.add(ds.get(i).id());
                }
            }
            if (!W.isEmpty()) ep = W.get(0).id();
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    public List<DistHit> knn(List<Float> q, int k, int ef, DistanceFn dist) {
        if (entryPt == -1) return Collections.emptyList();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<DistHit> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id();
            }
        }
        List<DistHit> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        return W.size() > k ? new ArrayList<>(W.subList(0, k)) : W;
    }

    public void remove(int id) {
        if (!graph.containsKey(id)) return;
        for (Node nd : graph.values())
            for (List<Integer> layer : nd.nbrs)
                layer.remove(Integer.valueOf(id));
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : graph.keySet())
                if (nid != id) { entryPt = nid; break; }
        }
        graph.remove(id);
    }

    public int size() { return graph.size(); }

    public static class GraphInfo {
        public int topLayer, nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();
        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();

        public static class NodeView {
            public int id, maxLyr;
            public String metadata, category;
        }
        public static class EdgeView {
            public int src, dst, lyr;
        }
    }

    public GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer  = topLayer;
        gi.nodeCount = graph.size();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) {
            gi.nodesPerLayer.add(0);
            gi.edgesPerLayer.add(0);
        }
        for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
            int  id = entry.getKey();
            Node nd = entry.getValue();

            GraphInfo.NodeView nv = new GraphInfo.NodeView();
            nv.id = id; nv.metadata = nd.item.metadata;
            nv.category = nd.item.category; nv.maxLyr = nd.maxLyr;
            gi.nodes.add(nv);

            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            GraphInfo.EdgeView ev = new GraphInfo.EdgeView();
                            ev.src = id; ev.dst = nid; ev.lyr = lc;
                            gi.edges.add(ev);
                        }
                    }
                }
            }
        }
        return gi;
    }
}
