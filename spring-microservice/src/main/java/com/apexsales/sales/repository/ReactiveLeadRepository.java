package com.apexsales.sales.repository;

import com.apexsales.sales.model.Lead;
import com.apexsales.sales.model.Interaction;
import com.apexsales.sales.model.CallAnalytics;
import org.springframework.stereotype.Repository;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.List;
import java.util.Map;

@Repository
public class ReactiveLeadRepository {

    private final Map<String, Lead> leads = new ConcurrentHashMap<>();
    private final Map<String, List<Interaction>> interactions = new ConcurrentHashMap<>();
    private final List<CallAnalytics> analyticsList = new CopyOnWriteArrayList<>();

    public Mono<Lead> saveLead(Lead lead) {
        if (lead.getLeadId() == null || lead.getLeadId().isEmpty()) {
            lead.setLeadId("lead_" + System.currentTimeMillis());
        }
        leads.put(lead.getLeadId(), lead);
        return Mono.just(lead);
    }

    public Mono<Lead> findLeadById(String leadId) {
        Lead lead = leads.get(leadId);
        return lead != null ? Mono.just(lead) : Mono.empty();
    }

    public Flux<Lead> findAllLeads() {
        return Flux.fromIterable(leads.values());
    }

    public Mono<Interaction> saveInteraction(Interaction interaction) {
        if (interaction.getId() == null) {
            interaction.setId("int_" + System.currentTimeMillis());
        }
        interactions.computeIfAbsent(interaction.getLeadId(), k -> new CopyOnWriteArrayList<>()).add(interaction);
        return Mono.just(interaction);
    }

    public Flux<Interaction> findInteractionsByLeadId(String leadId) {
        List<Interaction> list = interactions.getOrDefault(leadId, List.of());
        return Flux.fromIterable(list);
    }

    public Mono<CallAnalytics> saveAnalytics(CallAnalytics analytics) {
        if (analytics.getId() == null) {
            analytics.setId("ana_" + System.currentTimeMillis());
        }
        analyticsList.add(analytics);
        return Mono.just(analytics);
    }

    public Flux<CallAnalytics> findAllAnalytics() {
        return Flux.fromIterable(analyticsList);
    }
}

