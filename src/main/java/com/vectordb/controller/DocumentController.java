package com.vectordb.controller;

import com.vectordb.db.DocumentDB;
import com.vectordb.model.DocItem;
import com.vectordb.service.OllamaClient;
import com.vectordb.service.TextChunker;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/doc")
@CrossOrigin(origins = "*")
public class DocumentController {

    private final DocumentDB   documentDB;
    private final OllamaClient ollama;
    private final TextChunker  chunker;

    public DocumentController(DocumentDB documentDB, OllamaClient ollama, TextChunker chunker) {
        this.documentDB = documentDB;
        this.ollama     = ollama;
        this.chunker    = chunker;
    }

    @PostMapping("/insert")
    public ResponseEntity<Map<String, Object>> insert(@RequestBody Map<String, Object> body) {
        String title = (String) body.get("title");
        String text  = (String) body.get("text");
        if (title == null || title.isBlank() || text == null || text.isBlank())
            return ResponseEntity.badRequest().body(Map.of("error", "need title and text"));

        List<String> chunks = chunker.chunk(text, 250, 30);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            List<Float> emb = ollama.embed(chunks.get(i));
            if (emb.isEmpty()) {
                return ResponseEntity.ok(Map.of("error",
                        "Ollama unavailable. Install from https://ollama.com then run: " +
                        "ollama pull nomic-embed-text && ollama pull llama3.2"));
            }
            String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
            ids.add(documentDB.insert(chunkTitle, chunks.get(i), emb));
        }

        return ResponseEntity.ok(Map.of(
                "ids",    ids,
                "chunks", chunks.size(),
                "dims",   documentDB.getDims()
        ));
    }

    @DeleteMapping("/delete/{id}")
    public ResponseEntity<Map<String, Object>> delete(@PathVariable int id) {
        return ResponseEntity.ok(Map.of("ok", documentDB.remove(id)));
    }

    @GetMapping("/list")
    public ResponseEntity<List<Map<String, Object>>> list() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (DocItem doc : documentDB.all()) {
            String preview = doc.text.length() > 120
                    ? doc.text.substring(0, 120) + "\u2026"
                    : doc.text;
            int words = doc.text.trim().split("\\s+").length;
            Map<String, Object> d = new LinkedHashMap<>();
            d.put("id",      doc.id);
            d.put("title",   doc.title);
            d.put("preview", preview);
            d.put("words",   words);
            result.add(d);
        }
        return ResponseEntity.ok(result);
    }

    @PostMapping("/search")
    public ResponseEntity<Map<String, Object>> search(@RequestBody Map<String, Object> body) {
        String question = (String) body.get("question");
        int k = body.containsKey("k") ? ((Number) body.get("k")).intValue() : 3;
        if (question == null || question.isBlank())
            return ResponseEntity.badRequest().body(Map.of("error", "need question"));

        List<Float> qEmb = ollama.embed(question);
        if (qEmb.isEmpty())
            return ResponseEntity.ok(Map.of("error", "Ollama unavailable"));

        List<DocumentDB.DocHit> hits = documentDB.search(qEmb, k);
        List<Map<String, Object>> contexts = new ArrayList<>();
        for (DocumentDB.DocHit h : hits) {
            contexts.add(Map.of(
                    "id",       h.doc().id,
                    "title",    h.doc().title,
                    "distance", h.dist()
            ));
        }
        return ResponseEntity.ok(Map.of("contexts", contexts));
    }

    @PostMapping("/ask")
    public ResponseEntity<Map<String, Object>> ask(@RequestBody Map<String, Object> body) {
        String question = (String) body.get("question");
        int k = body.containsKey("k") ? ((Number) body.get("k")).intValue() : 3;
        if (question == null || question.isBlank())
            return ResponseEntity.badRequest().body(Map.of("error", "need question"));

        List<Float> qEmb = ollama.embed(question);
        if (qEmb.isEmpty())
            return ResponseEntity.ok(Map.of("error", "Ollama unavailable"));

        List<DocumentDB.DocHit> hits = documentDB.search(qEmb, k);

        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append("[").append(i + 1).append("] ")
               .append(hits.get(i).doc().title).append(":\n")
               .append(hits.get(i).doc().text).append("\n\n");
        }
        String prompt =
                "You are a helpful assistant. The user has provided their personal notes as context below. " +
                "Prioritize answering from the notes. If the notes contain relevant information, use it directly. " +
                "If the notes do not cover the topic, answer from your general knowledge and mention it briefly.\n\n" +
                "Notes/Context:\n" + ctx +
                "Question: " + question + "\n\nAnswer:";

        String answer = ollama.generate(prompt);

        List<Map<String, Object>> contexts = new ArrayList<>();
        for (DocumentDB.DocHit h : hits) {
            Map<String, Object> c = new LinkedHashMap<>();
            c.put("id",       h.doc().id);
            c.put("title",    h.doc().title);
            c.put("text",     h.doc().text);
            c.put("distance", h.dist());
            contexts.add(c);
        }

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("answer",   answer);
        resp.put("model",    ollama.genModel);
        resp.put("contexts", contexts);
        resp.put("docCount", documentDB.size());
        return ResponseEntity.ok(resp);
    }
}
