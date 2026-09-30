package com.vintushtech.sales.controller;

import com.vintushtech.sales.model.Lead;
import com.vintushtech.sales.model.Interaction;
import com.vintushtech.sales.model.CallAnalytics;
import com.vintushtech.sales.service.LeadPersistenceService;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class LeadController {

    private final LeadPersistenceService persistenceService;

    public LeadController(LeadPersistenceService persistenceService) {
        this.persistenceService = persistenceService;
    }

    @GetMapping("/health")
    public Mono<Map<String, String>> health() {
        return Mono.just(Map.of(
                "status", "UP",
                "service", "VintushTech Reactive Spring WebFlux Microservice",
                "framework", "Spring Boot 3 + Project Reactor"
        ));
    }

    @PostMapping("/leads")
    @ResponseStatus(HttpStatus.CREATED)
    public Mono<Lead> createOrUpdateLead(@RequestBody Lead lead) {
        return persistenceService.recordOrUpdateLead(lead);
    }

    @GetMapping("/leads")
    public Flux<Lead> getAllLeads() {
        return persistenceService.getAllLeads();
    }

    @GetMapping("/leads/{id}")
    public Mono<Lead> getLeadById(@PathVariable("id") String id) {
        return persistenceService.getLead(id);
    }

    @PostMapping("/leads/{id}/interactions")
    @ResponseStatus(HttpStatus.CREATED)
    public Mono<Interaction> addInteraction(@PathVariable("id") String leadId, @RequestBody Interaction interaction) {
        interaction.setLeadId(leadId);
        return persistenceService.recordInteraction(interaction);
    }

    @GetMapping("/leads/{id}/interactions")
    public Flux<Interaction> getInteractions(@PathVariable("id") String leadId) {
        return persistenceService.getInteractions(leadId);
    }

    @PostMapping("/analytics")
    @ResponseStatus(HttpStatus.CREATED)
    public Mono<CallAnalytics> saveAnalytics(@RequestBody CallAnalytics analytics) {
        return persistenceService.recordAnalytics(analytics);
    }

    @GetMapping("/analytics")
    public Flux<CallAnalytics> getAllAnalytics() {
        return persistenceService.getAllAnalytics();
    }
}
