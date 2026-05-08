package com.vectordb.controller;

import com.vectordb.algo.Distances;
import com.vectordb.db.VectorDB;
import com.vectordb.model.VectorItem;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@CrossOrigin(origins = "*")
public class VectorController {

    private final VectorDB vectorDB;

    public VectorController(VectorDB vectorDB) {
        this.vectorDB = vectorDB;
    }

    @GetMapping("/search")
    public ResponseEntity<Map<String, Object>> search(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric,
            @RequestParam(defaultValue = "hnsw") String algo) {

        List<Float> q = parseVec(v);
        if (q.size() != VectorDB.DIMS)
            return ResponseEntity.badRequest().body(Map.of("error", "need " + VectorDB.DIMS + "D vector"));

        VectorDB.SearchOut out = vectorDB.search(q, k, metric, algo);
        List<Map<String, Object>> results = new ArrayList<>();
        for (VectorDB.Hit h : out.hits) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("id",        h.id);
            r.put("metadata",  h.meta);
            r.put("category",  h.cat);
            r.put("distance",  h.dist);
            r.put("embedding", h.emb);
            results.add(r);
        }
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("results",   results);
        resp.put("latencyUs", out.latencyUs);
        resp.put("algo",      out.algo);
        resp.put("metric",    out.metric);
        return ResponseEntity.ok(resp);
    }

    @PostMapping("/insert")
    public ResponseEntity<Map<String, Object>> insert(@RequestBody Map<String, Object> body) {
        String meta = (String) body.get("metadata");
        String cat  = (String) body.getOrDefault("category", "default");
        @SuppressWarnings("unchecked")
        List<Number> embRaw = (List<Number>) body.get("embedding");

        if (meta == null || embRaw == null)
            return ResponseEntity.badRequest().body(Map.of("error", "invalid body"));

        List<Float> emb = embRaw.stream().map(Number::floatValue).collect(Collectors.toList());
        if (emb.size() != VectorDB.DIMS)
            return ResponseEntity.badRequest().body(Map.of("error", "invalid body"));

        int id = vectorDB.insert(meta, cat, emb, Distances::cosine);
        return ResponseEntity.ok(Map.of("id", id));
    }

    @DeleteMapping("/delete/{id}")
    public ResponseEntity<Map<String, Object>> delete(@PathVariable int id) {
        return ResponseEntity.ok(Map.of("ok", vectorDB.remove(id)));
    }

    @GetMapping("/items")
    public ResponseEntity<List<Map<String, Object>>> items() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (VectorItem v : vectorDB.all()) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id",        v.id);
            item.put("metadata",  v.metadata);
            item.put("category",  v.category);
            item.put("embedding", v.emb);
            result.add(item);
        }
        return ResponseEntity.ok(result);
    }

    @GetMapping("/benchmark")
    public ResponseEntity<Map<String, Object>> benchmark(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric) {

        List<Float> q = parseVec(v);
        if (q.size() != VectorDB.DIMS)
            return ResponseEntity.badRequest().body(Map.of("error", "need " + VectorDB.DIMS + "D vector"));

        VectorDB.BenchOut b = vectorDB.benchmark(q, k, metric);
        return ResponseEntity.ok(Map.of(
                "bruteforceUs", b.bfUs,
                "kdtreeUs",     b.kdUs,
                "hnswUs",       b.hnswUs,
                "itemCount",    b.itemCount
        ));
    }

    @GetMapping("/hnsw-info")
    public ResponseEntity<?> hnswInfo() {
        return ResponseEntity.ok(vectorDB.hnswInfo());
    }

    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> stats() {
        return ResponseEntity.ok(Map.of(
                "count",      vectorDB.size(),
                "dims",       VectorDB.DIMS,
                "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                "metrics",    List.of("euclidean", "cosine", "manhattan")
        ));
    }

    private List<Float> parseVec(String s) {
        List<Float> v = new ArrayList<>();
        for (String t : s.split(","))
            try { v.add(Float.parseFloat(t.trim())); } catch (NumberFormatException ignored) {}
        return v;
    }
}
