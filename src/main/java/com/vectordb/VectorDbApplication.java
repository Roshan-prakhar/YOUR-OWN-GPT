package com.vectordb;

import com.vectordb.algo.DistanceFn;
import com.vectordb.algo.Distances;
import com.vectordb.db.VectorDB;
import com.vectordb.service.OllamaClient;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import java.util.List;

@SpringBootApplication
public class VectorDbApplication {

    public static void main(String[] args) {
        SpringApplication.run(VectorDbApplication.class, args);
    }

    @Bean
    CommandLineRunner init(VectorDB db, OllamaClient ollama) {
        return args -> {
            loadDemo(db);
            boolean ollamaUp = ollama.isAvailable();
            System.out.println("=== VectorDB Engine (Java) ===");
            System.out.println("http://localhost:8080");
            System.out.println(db.size() + " demo vectors | " + VectorDB.DIMS + " dims | HNSW+KD-Tree+BruteForce");
            System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
            if (ollamaUp) System.out.println("  embed: " + ollama.embedModel + "  gen: " + ollama.genModel);
        };
    }

    private void loadDemo(VectorDB db) {
        DistanceFn d = Distances::cosine;
        db.insert("Linked List: nodes connected by pointers", "cs",
            List.of(0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f), d);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            List.of(0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f), d);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            List.of(0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f), d);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            List.of(0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f), d);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            List.of(0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f), d);
        db.insert("Calculus: derivatives integrals and limits", "math",
            List.of(0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f), d);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            List.of(0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f), d);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
            List.of(0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f), d);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            List.of(0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f), d);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
            List.of(0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f), d);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            List.of(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f), d);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            List.of(0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f), d);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            List.of(0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f), d);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            List.of(0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f), d);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            List.of(0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f), d);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            List.of(0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f), d);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            List.of(0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f), d);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            List.of(0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f), d);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            List.of(0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f), d);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            List.of(0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f), d);
    }
}
