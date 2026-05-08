package com.vectordb.controller;

import com.vectordb.db.DocumentDB;
import com.vectordb.db.VectorDB;
import com.vectordb.service.OllamaClient;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@RestController
@CrossOrigin(origins = "*")
public class StatusController {

    private final VectorDB   vectorDB;
    private final DocumentDB documentDB;
    private final OllamaClient ollama;

    public StatusController(VectorDB vectorDB, DocumentDB documentDB, OllamaClient ollama) {
        this.vectorDB   = vectorDB;
        this.documentDB = documentDB;
        this.ollama     = ollama;
    }

    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> status() {
        boolean up = ollama.isAvailable();
        return ResponseEntity.ok(Map.of(
                "ollamaAvailable", up,
                "embedModel",      ollama.embedModel,
                "genModel",        ollama.genModel,
                "docCount",        documentDB.size(),
                "docDims",         documentDB.getDims(),
                "demoDims",        VectorDB.DIMS,
                "demoCount",       vectorDB.size()
        ));
    }

    @GetMapping("/")
    public ResponseEntity<String> index() throws IOException {
        for (String candidate : new String[]{"index.html", "../Your-OWN-AI-main/index.html"}) {
            Path p = Paths.get(candidate);
            if (Files.exists(p)) {
                return ResponseEntity.ok()
                        .contentType(MediaType.TEXT_HTML)
                        .body(Files.readString(p));
            }
        }
        return ResponseEntity.notFound().build();
    }
}
