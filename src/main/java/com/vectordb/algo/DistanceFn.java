package com.vectordb.algo;

import java.util.List;

@FunctionalInterface
public interface DistanceFn {
    float apply(List<Float> a, List<Float> b);
}
