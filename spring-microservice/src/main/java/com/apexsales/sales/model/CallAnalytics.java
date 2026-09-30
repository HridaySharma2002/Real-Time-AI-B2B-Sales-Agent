package com.apexsales.sales.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.time.Instant;
import java.util.Map;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class CallAnalytics {
    private String id;
    private String callId;
    private String sessionId;
    private Double durationSec;
    private Double dropOffTurn;
    private Double sentimentScore;
    private Integer interruptionCount;
    private String dominantFriction;
    private Boolean demoBooked;
    private Integer assignedCluster;
    private String diagnosis;
    private String timestamp;
    private Map<String, Object> extraMetrics;

    public CallAnalytics() {
        this.timestamp = Instant.now().toString();
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getCallId() { return callId; }
    public void setCallId(String callId) { this.callId = callId; }

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }

    public Double getDurationSec() { return durationSec; }
    public void setDurationSec(Double durationSec) { this.durationSec = durationSec; }

    public Double getDropOffTurn() { return dropOffTurn; }
    public void setDropOffTurn(Double dropOffTurn) { this.dropOffTurn = dropOffTurn; }

    public Double getSentimentScore() { return sentimentScore; }
    public void setSentimentScore(Double sentimentScore) { this.sentimentScore = sentimentScore; }

    public Integer getInterruptionCount() { return interruptionCount; }
    public void setInterruptionCount(Integer interruptionCount) { this.interruptionCount = interruptionCount; }

    public String getDominantFriction() { return dominantFriction; }
    public void setDominantFriction(String dominantFriction) { this.dominantFriction = dominantFriction; }

    public Boolean getDemoBooked() { return demoBooked; }
    public void setDemoBooked(Boolean demoBooked) { this.demoBooked = demoBooked; }

    public Integer getAssignedCluster() { return assignedCluster; }
    public void setAssignedCluster(Integer assignedCluster) { this.assignedCluster = assignedCluster; }

    public String getDiagnosis() { return diagnosis; }
    public void setDiagnosis(String diagnosis) { this.diagnosis = diagnosis; }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

    public Map<String, Object> getExtraMetrics() { return extraMetrics; }
    public void setExtraMetrics(Map<String, Object> extraMetrics) { this.extraMetrics = extraMetrics; }
}

