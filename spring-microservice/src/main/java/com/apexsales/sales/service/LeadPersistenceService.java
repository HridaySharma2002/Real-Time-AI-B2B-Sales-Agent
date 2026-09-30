package com.apexsales.sales.service;

import com.apexsales.sales.client.FastApiWebClient;
import com.apexsales.sales.model.Lead;
import com.apexsales.sales.model.Interaction;
import com.apexsales.sales.model.CallAnalytics;
import com.apexsales.sales.repository.ReactiveLeadRepository;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;

@Service
public class LeadPersistenceService {

    private final ReactiveLeadRepository repository;
    private final FastApiWebClient fastApiClient;

    public LeadPersistenceService(ReactiveLeadRepository repository, FastApiWebClient fastApiClient) {
        this.repository = repository;
        this.fastApiClient = fastApiClient;
    }

    public Mono<Lead> recordOrUpdateLead(Lead lead) {
        lead.setUpdatedAt(Instant.now().toString());

        // Reactive async enrichment from FastAPI if company is provided
        if (lead.getCompany() != null && !lead.getCompany().isEmpty() && lead.getPersona() == null) {
            return fastApiClient.fetchEnrichment(lead.getCompany())
                    .flatMap(enrichment -> {
                        if (enrichment != null && enrichment.containsKey("recommended_tier")) {
                            lead.setEstimatedValue(String.valueOf(enrichment.get("recommended_tier")));
                        }
                        return repository.saveLead(lead);
                    })
                    .defaultIfEmpty(lead)
                    .flatMap(repository::saveLead);
        }

        return repository.saveLead(lead);
    }

    public Mono<Lead> getLead(String leadId) {
        return repository.findLeadById(leadId);
    }

    public Flux<Lead> getAllLeads() {
        return repository.findAllLeads();
    }

    public Mono<Interaction> recordInteraction(Interaction interaction) {
        return repository.saveInteraction(interaction);
    }

    public Flux<Interaction> getInteractions(String leadId) {
        return repository.findInteractionsByLeadId(leadId);
    }

    public Mono<CallAnalytics> recordAnalytics(CallAnalytics analytics) {
        return repository.saveAnalytics(analytics);
    }

    public Flux<CallAnalytics> getAllAnalytics() {
        return repository.findAllAnalytics();
    }
}

