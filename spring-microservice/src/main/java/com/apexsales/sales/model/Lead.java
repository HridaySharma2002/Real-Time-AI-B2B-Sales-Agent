package com.apexsales.sales.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.time.Instant;
import java.util.Map;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class Lead {
    private String leadId;
    private String company;
    private String contactName;
    private String email;
    private String phone;
    private String persona;
    private Integer clusterId;
    private String frictionTopic;
    private String status; // QUALIFIED, NURTURING, CLOSED_WON, UNQUALIFIED
    private String estimatedValue;
    private String lastInteraction;
    private String createdAt;
    private String updatedAt;
    private Map<String, Object> metadata;

    public Lead() {
        this.createdAt = Instant.now().toString();
        this.updatedAt = Instant.now().toString();
        this.status = "DISCOVERY";
    }

    public Lead(String leadId, String company) {
        this();
        this.leadId = leadId;
        this.company = company;
    }

    public String getLeadId() { return leadId; }
    public void setLeadId(String leadId) { this.leadId = leadId; }

    public String getCompany() { return company; }
    public void setCompany(String company) { this.company = company; }

    public String getContactName() { return contactName; }
    public void setContactName(String contactName) { this.contactName = contactName; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }

    public String getPersona() { return persona; }
    public void setPersona(String persona) { this.persona = persona; }

    public Integer getClusterId() { return clusterId; }
    public void setClusterId(Integer clusterId) { this.clusterId = clusterId; }

    public String getFrictionTopic() { return frictionTopic; }
    public void setFrictionTopic(String frictionTopic) { this.frictionTopic = frictionTopic; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getEstimatedValue() { return estimatedValue; }
    public void setEstimatedValue(String estimatedValue) { this.estimatedValue = estimatedValue; }

    public String getLastInteraction() { return lastInteraction; }
    public void setLastInteraction(String lastInteraction) { this.lastInteraction = lastInteraction; }

    public String getCreatedAt() { return createdAt; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }

    public String getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(String updatedAt) { this.updatedAt = updatedAt; }

    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
}

