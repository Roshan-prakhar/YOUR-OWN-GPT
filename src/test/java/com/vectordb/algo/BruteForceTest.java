package com.vectordb.algo;

import com.vectordb.model.VectorItem;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Brute Force Exact KNN Retrieval Tests")
public class BruteForceTest {

    private BruteForce bruteForce;
    private static final float EPSILON = 1e-5f;

    @BeforeEach
    void setUp() {
        bruteForce = new BruteForce();
    }

    @Test
    @DisplayName("BruteForce Manhattan KNN: Exact distance calculation and ascending rank order")
    void testManhattanKnnOrder() {
        // Points in 2D space:
        // Item 1: (1, 1) -> Manhattan dist to (0,0) = 2.0
        // Item 2: (2, 2) -> Manhattan dist to (0,0) = 4.0
        // Item 3: (0.5, 0.5) -> Manhattan dist to (0,0) = 1.0
        // Item 4: (3, 0) -> Manhattan dist to (0,0) = 3.0
        // Item 5: (-0.2, -0.3) -> Manhattan dist to (0,0) = 0.5

        bruteForce.insert(new VectorItem(1, "Item1", "catA", List.of(1.0f, 1.0f)));
        bruteForce.insert(new VectorItem(2, "Item2", "catB", List.of(2.0f, 2.0f)));
        bruteForce.insert(new VectorItem(3, "Item3", "catA", List.of(0.5f, 0.5f)));
        bruteForce.insert(new VectorItem(4, "Item4", "catC", List.of(3.0f, 0.0f)));
        bruteForce.insert(new VectorItem(5, "Item5", "catC", List.of(-0.2f, -0.3f)));

        List<Float> query = List.of(0.0f, 0.0f);
        List<DistHit> hits = bruteForce.knn(query, 3, Distances::manhattan);

        assertEquals(3, hits.size(), "Should retrieve exactly k=3 nearest neighbors");

        // Rank 1: Item 5 with dist = 0.5
        assertEquals(5, hits.get(0).id());
        assertEquals(0.5f, hits.get(0).dist(), EPSILON);

        // Rank 2: Item 3 with dist = 1.0
        assertEquals(3, hits.get(1).id());
        assertEquals(1.0f, hits.get(1).dist(), EPSILON);

        // Rank 3: Item 1 with dist = 2.0
        assertEquals(1, hits.get(2).id());
        assertEquals(2.0f, hits.get(2).dist(), EPSILON);

        // Verify strictly non-decreasing distances
        for (int i = 0; i < hits.size() - 1; i++) {
            assertTrue(hits.get(i).dist() <= hits.get(i + 1).dist(),
                    "Hits must be sorted in non-decreasing order of distance");
        }
    }

    @Test
    @DisplayName("BruteForce Manhattan: Remove item updates search results")
    void testRemoveItem() {
        bruteForce.insert(new VectorItem(1, "Item1", "catA", List.of(0.1f, 0.1f)));
        bruteForce.insert(new VectorItem(2, "Item2", "catA", List.of(0.5f, 0.5f)));

        List<DistHit> hitsBefore = bruteForce.knn(List.of(0.0f, 0.0f), 1, Distances::manhattan);
        assertEquals(1, hitsBefore.get(0).id());

        bruteForce.remove(1);
        List<DistHit> hitsAfter = bruteForce.knn(List.of(0.0f, 0.0f), 1, Distances::manhattan);
        assertEquals(2, hitsAfter.get(0).id());
    }

    @Test
    @DisplayName("BruteForce: k larger than collection size returns all sorted items")
    void testKLargerThanSize() {
        bruteForce.insert(new VectorItem(1, "A", "c", List.of(1.0f, 2.0f)));
        bruteForce.insert(new VectorItem(2, "B", "c", List.of(3.0f, 4.0f)));

        List<DistHit> hits = bruteForce.knn(List.of(0.0f, 0.0f), 10, Distances::manhattan);
        assertEquals(2, hits.size());
    }
}
