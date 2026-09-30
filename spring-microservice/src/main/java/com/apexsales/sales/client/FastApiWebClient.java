package com.apexsales.sales.client;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Component
public class FastApiWebClient {

    private final WebClient webClient;

    public FastApiWebClient(@Value("${fastapi.service.url:http://localhost:8000}") String fastApiBaseUrl) {
        this.webClient = WebClient.builder()
                .baseUrl(fastApiBaseUrl)
                .build();
    }

    /**
     * Non-blocking call to FastAPI B2B enrichment endpoint.
     */
    public Mono<Map> fetchEnrichment(String companyQuery) {
        return this.webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/api/b2b/enrich")
                        .queryParam("query", companyQuery)
                        .build())
                .retrieve()
                .bodyToMono(Map.class)
                .onErrorResume(e -> Mono.empty());
    }

    /**
     * Non-blocking call to check FastAPI health.
     */
    public Mono<Map> checkFastApiHealth() {
        return this.webClient.get()
                .uri("/")
                .retrieve()
                .bodyToMono(Map.class)
                .onErrorResume(e -> Mono.just(Map.of("status", "fastapi_unreachable")));
    }
}

