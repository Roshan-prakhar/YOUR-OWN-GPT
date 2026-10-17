package com.vectordb.controller;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@DisplayName("VectorController REST API & Evaluation Integration Tests")
public class VectorControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    @DisplayName("GET /evaluate returns HNSW vs BruteForce evaluation using Manhattan distance")
    void testEvaluateEndpointManhattan() throws Exception {
        // 16D CS query vector
        String vec = "0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.1,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06";

        mockMvc.perform(get("/evaluate")
                        .param("v", vec)
                        .param("k", "5")
                        .param("metric", "manhattan")
                        .param("ef", "50"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.metric", is("manhattan")))
                .andExpect(jsonPath("$.k", is(5)))
                .andExpect(jsonPath("$.recallAtK", is(1.0)))
                .andExpect(jsonPath("$.precisionAtK", is(1.0)))
                .andExpect(jsonPath("$.top1Match", is(true)))
                .andExpect(jsonPath("$.distanceCalculationsValid", is(true)))
                .andExpect(jsonPath("$.hnswRetrieval", hasSize(5)))
                .andExpect(jsonPath("$.groundTruth", hasSize(5)))
                .andExpect(jsonPath("$.hnswRetrieval[0].distance", closeTo(0.0, 0.001)));
    }

    @Test
    @DisplayName("GET /search supports Manhattan distance metric")
    void testSearchEndpointManhattan() throws Exception {
        String vec = "0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.1,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06";

        mockMvc.perform(get("/search")
                        .param("v", vec)
                        .param("k", "3")
                        .param("metric", "manhattan")
                        .param("algo", "bruteforce"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.metric", is("manhattan")))
                .andExpect(jsonPath("$.algo", is("bruteforce")))
                .andExpect(jsonPath("$.results", hasSize(3)))
                .andExpect(jsonPath("$.results[0].category", is("cs")));
    }
}
