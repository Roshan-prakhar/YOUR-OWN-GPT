package com.vectordb.algo;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Distance Metrics Unit Tests")
public class DistancesTest {

    private static final float EPSILON = 1e-5f;

    @Test
    @DisplayName("Manhattan Distance: Manual step-by-step verification on 3D vectors")
    void testManhattanStepByStep() {
        // A = [1.0, 2.0, 3.0], B = [4.0, 0.0, 7.0]
        // |1.0 - 4.0| = 3.0
        // |2.0 - 0.0| = 2.0
        // |3.0 - 7.0| = 4.0
        // Sum = 3.0 + 2.0 + 4.0 = 9.0
        List<Float> a = List.of(1.0f, 2.0f, 3.0f);
        List<Float> b = List.of(4.0f, 0.0f, 7.0f);

        float dist = Distances.manhattan(a, b);
        assertEquals(9.0f, dist, EPSILON, "Manhattan distance should be 9.0");
    }

    @Test
    @DisplayName("Manhattan Distance: Identity of indiscernibles (dist(x, x) == 0)")
    void testManhattanZeroDistance() {
        List<Float> a = List.of(0.85f, 0.12f, -0.45f, 1.20f);
        float dist = Distances.manhattan(a, a);
        assertEquals(0.0f, dist, EPSILON, "Distance to self must be 0");
    }

    @Test
    @DisplayName("Manhattan Distance: Symmetry (dist(a, b) == dist(b, a))")
    void testManhattanSymmetry() {
        List<Float> a = List.of(0.1f, 0.5f, 0.9f, -0.2f);
        List<Float> b = List.of(0.4f, -0.3f, 0.2f, 0.8f);

        float d_ab = Distances.manhattan(a, b);
        float d_ba = Distances.manhattan(b, a);
        assertEquals(d_ab, d_ba, EPSILON, "Manhattan distance must be symmetric");
    }

    @Test
    @DisplayName("Manhattan Distance: Triangle inequality (dist(a, c) <= dist(a, b) + dist(b, c))")
    void testManhattanTriangleInequality() {
        List<Float> a = List.of(1.0f, 2.0f, 3.0f);
        List<Float> b = List.of(4.0f, 5.0f, 6.0f);
        List<Float> c = List.of(2.0f, 0.0f, -1.0f);

        float d_ac = Distances.manhattan(a, c);
        float d_ab = Distances.manhattan(a, b);
        float d_bc = Distances.manhattan(b, c);

        assertTrue(d_ac <= d_ab + d_bc + EPSILON,
                "Triangle inequality violated: " + d_ac + " > " + (d_ab + d_bc));
    }

    @Test
    @DisplayName("Manhattan Distance: Handling negative and mixed values")
    void testManhattanNegativeValues() {
        // A = [-1.5, 3.2, -4.0], B = [2.5, -1.8, 1.0]
        // |-1.5 - 2.5|  = 4.0
        // |3.2 - (-1.8)| = 5.0
        // |-4.0 - 1.0|  = 5.0
        // Total = 14.0
        List<Float> a = List.of(-1.5f, 3.2f, -4.0f);
        List<Float> b = List.of(2.5f, -1.8f, 1.0f);

        float dist = Distances.manhattan(a, b);
        assertEquals(14.0f, dist, EPSILON, "Manhattan distance with negative coordinates must be 14.0");
    }

    @Test
    @DisplayName("Compare Manhattan vs Euclidean vs Cosine on identical pair")
    void testCompareDistanceMetrics() {
        List<Float> a = List.of(1.0f, 2.0f, 3.0f);
        List<Float> b = List.of(4.0f, 0.0f, 7.0f);

        float manhattan = Distances.manhattan(a, b);
        float euclidean = Distances.euclidean(a, b);
        float cosine = Distances.cosine(a, b);

        // Manhattan: |3| + |2| + |4| = 9.0
        // Euclidean: sqrt(3^2 + 2^2 + 4^2) = sqrt(29) ≈ 5.385165
        // Cosine: 1 - (4 + 0 + 21) / (sqrt(14) * sqrt(65)) = 1 - 25 / sqrt(910) ≈ 1 - 0.82876 = 0.17124
        assertEquals(9.0f, manhattan, EPSILON);
        assertEquals((float) Math.sqrt(29), euclidean, EPSILON);
        assertTrue(euclidean < manhattan, "Euclidean distance (L2 norm) is always <= Manhattan distance (L1 norm)");
        assertTrue(cosine >= 0.0f && cosine <= 2.0f, "Cosine distance is within valid range [0, 2]");
    }

    @Test
    @DisplayName("Verify Distances.get factory method resolves properly")
    void testFactoryGet() {
        DistanceFn manhattanFn = Distances.get("manhattan");
        DistanceFn euclideanFn = Distances.get("euclidean");
        DistanceFn cosineFn = Distances.get("cosine");
        DistanceFn defaultFn = Distances.get("unknown");

        List<Float> a = List.of(1.0f, 2.0f);
        List<Float> b = List.of(4.0f, 6.0f);

        assertEquals(7.0f, manhattanFn.apply(a, b), EPSILON); // |3| + |4| = 7
        assertEquals(5.0f, euclideanFn.apply(a, b), EPSILON); // sqrt(9 + 16) = 5
        assertEquals(5.0f, defaultFn.apply(a, b), EPSILON);   // default is euclidean
    }
}
