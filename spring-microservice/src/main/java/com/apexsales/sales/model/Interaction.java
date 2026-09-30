package com.apexsales.sales.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.time.Instant;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class Interaction {
    private String id;
    private String leadId;
    private String sessionId;
    private String userMsg;
    private String aiReply;
    private Double sentiment;
    private String frictionTopic;
    private String persona;
    private Integer clusterId;
    private String timestamp;

    public Interaction() {
        this.timestamp = Instant.now().toString();
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getLeadId() { return leadId; }
    public void setLeadId(String leadId) { this.leadId = leadId; }

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }

    public String getUserMsg() { return userMsg; }
    public void setUserMsg(String userMsg) { this.userMsg = userMsg; }

    public String getAiReply() { return aiReply; }
    public void setAiReply(String aiReply) { this.aiReply = aiReply; }

    public Double getSentiment() { return sentiment; }
    public void setSentiment(Double sentiment) { this.sentiment = sentiment; }

    public String getFrictionTopic() { return frictionTopic; }
    public void setFrictionTopic(String frictionTopic) { this.frictionTopic = frictionTopic; }

    public String getPersona() { return persona; }
    public void setPersona(String persona) { this.persona = persona; }

    public Integer getClusterId() { return clusterId; }
    public void setClusterId(Integer clusterId) { this.clusterId = clusterId; }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }
}

