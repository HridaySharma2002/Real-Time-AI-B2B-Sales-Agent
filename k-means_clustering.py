"""
k-means_clustering.py - Sales Call Clustering & Continuous RAG Optimization
Integrated with Rag_Knowledge_base.txt and Rag.py for Real-Time B2B Sales Agents.

Concept:
Using K-Means to cluster recorded sales calls based on:
1. Drop-off points (turn number/timestamp where prospect dropped off or objected)
2. Call duration (seconds)
3. Low-engagement & friction topics (e.g. pricing, competitors, complexity, trust)
4. Sentiment polarity score (-1.0 negative to +1.0 positive)
5. Agent latency & interruption counts

Outcome:
Mathematically pinpoints failure patterns in sales calls and automatically updates/enriches
`Rag_Knowledge_base.txt` with targeted objection-handling scripts to resolve weaknesses.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SalesCallClustering")

KB_FILE_PATH = os.path.join(os.path.dirname(__file__), "Rag_Knowledge_base.txt")


# =====================================================================
# Synthetic / Recorded Sales Call Data Generator
# =====================================================================

def generate_sample_sales_call_logs(num_calls: int = 150) -> pd.DataFrame:
    """
    Generates realistic recorded sales call metrics representing diverse customer interactions.
    Patterns:
    - Group A (Successful Close): Longer duration, positive sentiment, high turn count, demo booked.
    - Group B (Pricing Friction Drop-off): Early/mid drop-off around pricing, negative sentiment.
    - Group C (Feature Complexity Drop-off): Mid-call drop-off, low engagement on technical specs.
    - Group D (Immediate Gatekeeper Bounce): Very short duration (<45s), low turns.
    """
    np.random.seed(42)
    
    topics = ["pricing", "competitor_drift", "security_compliance", "technical_latency", "gatekeeper_bounce", "product_features"]
    
    records = []
    for i in range(num_calls):
        call_type = np.random.choice(["success", "pricing_drop", "tech_drop", "gatekeeper_bounce"], p=[0.30, 0.35, 0.20, 0.15])
        
        if call_type == "success":
            duration = np.random.normal(loc=320, scale=40)  # ~5.3 mins
            drop_off_turn = np.random.normal(loc=12, scale=2)
            sentiment = np.random.uniform(0.5, 0.95)
            topic = np.random.choice(["product_features", "pricing", "security_compliance"])
            interruptions = np.random.poisson(lam=1.2)
            demo_booked = 1
            agent_latency_ms = np.random.normal(loc=420, scale=30)
            
        elif call_type == "pricing_drop":
            duration = np.random.normal(loc=140, scale=25)  # ~2.3 mins
            drop_off_turn = np.random.normal(loc=4, scale=1)
            sentiment = np.random.uniform(-0.6, -0.1)
            topic = "pricing"
            interruptions = np.random.poisson(lam=3.5)
            demo_booked = 0
            agent_latency_ms = np.random.normal(loc=480, scale=40)
            
        elif call_type == "tech_drop":
            duration = np.random.normal(loc=190, scale=30)  # ~3.1 mins
            drop_off_turn = np.random.normal(loc=6, scale=1.5)
            sentiment = np.random.uniform(-0.4, 0.1)
            topic = np.random.choice(["competitor_drift", "technical_latency", "security_compliance"])
            interruptions = np.random.poisson(lam=2.8)
            demo_booked = 0
            agent_latency_ms = np.random.normal(loc=550, scale=60)
            
        else:  # gatekeeper_bounce
            duration = np.random.normal(loc=35, scale=10)   # ~35 secs
            drop_off_turn = np.random.normal(loc=2, scale=0.5)
            sentiment = np.random.uniform(-0.8, -0.3)
            topic = "gatekeeper_bounce"
            interruptions = np.random.poisson(lam=1.0)
            demo_booked = 0
            agent_latency_ms = np.random.normal(loc=410, scale=30)

        records.append({
            "call_id": f"CALL_{1000 + i}",
            "duration_sec": max(15.0, round(float(duration), 1)),
            "drop_off_turn": max(1.0, round(float(drop_off_turn), 1)),
            "sentiment_score": round(float(sentiment), 2),
            "interruption_count": max(0, int(interruptions)),
            "agent_latency_ms": max(200.0, round(float(agent_latency_ms), 1)),
            "friction_topic": topic,
            "demo_booked": int(demo_booked)
        })

    return pd.DataFrame(records)


# =====================================================================
# K-Means Call Analysis & Failure Pinpointer
# =====================================================================

class SalesCallClustering:
    """
    K-Means clustering engine to detect call failure patterns, drop-off points,
    and friction topics across recorded sales calls.
    """

    FEATURE_COLS = [
        "duration_sec",
        "drop_off_turn",
        "sentiment_score",
        "interruption_count",
        "agent_latency_ms"
    ]

    def __init__(self, k_clusters: int = 4):
        self.k_clusters = k_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        self.df: Optional[pd.DataFrame] = None
        self.cluster_summary: Dict[int, Dict[str, Any]] = {}

    def fit_and_analyze(self, call_data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs feature scaling, fits K-Means, and creates actionable cluster insights.
        """
        self.df = call_data.copy()
        
        # Scale numerical features
        x_scaled = self.scaler.fit_transform(self.df[self.FEATURE_COLS])
        
        # Fit K-Means
        self.df["cluster"] = self.kmeans.fit_predict(x_scaled)
        
        # Calculate silhouette score for cluster separation quality
        if len(self.df) > self.k_clusters:
            sil_score = silhouette_score(x_scaled, self.df["cluster"])
            logger.info(f"K-Means fitted with {self.k_clusters} clusters. Silhouette Score: {round(sil_score, 3)}")

        # Analyze failure profiles per cluster
        for c in range(self.k_clusters):
            c_df = self.df[self.df["cluster"] == c]
            total_in_cluster = len(c_df)
            demo_rate = c_df["demo_booked"].mean() * 100
            fail_rate = 100.0 - demo_rate
            
            top_topic = c_df["friction_topic"].mode()[0] if not c_df.empty else "unknown"
            avg_duration = c_df["duration_sec"].mean()
            avg_drop_turn = c_df["drop_off_turn"].mean()
            avg_sentiment = c_df["sentiment_score"].mean()

            # Categorize cluster diagnostic
            if demo_rate >= 70:
                diagnosis = "HIGH_CONVERTING_SUCCESS"
                severity = "LOW"
            elif avg_duration < 60:
                diagnosis = "EARLY_GATEKEEPER_BOUNCE"
                severity = "HIGH"
            elif "pricing" in top_topic or avg_sentiment < -0.2:
                diagnosis = "PRICING_OBJECTION_FRICTION"
                severity = "CRITICAL"
            else:
                diagnosis = "FEATURE_COMPLEXITY_DROP_OFF"
                severity = "MEDIUM"

            self.cluster_summary[c] = {
                "cluster_id": c,
                "call_count": total_in_cluster,
                "conversion_rate": round(demo_rate, 1),
                "failure_rate": round(fail_rate, 1),
                "dominant_topic": top_topic,
                "avg_duration_sec": round(avg_duration, 1),
                "avg_drop_turn": round(avg_drop_turn, 1),
                "avg_sentiment": round(avg_sentiment, 2),
                "diagnosis": diagnosis,
                "severity": severity
            }

        return self.df

    def print_diagnostics(self):
        """Displays formatted mathematical breakdown of call failure clusters."""
        print("\n" + "=" * 80)
        print("  MATHEMATICAL CALL FAILURE PINPOINTING (K-MEANS CLUSTERING)")
        print("=" * 80)
        print(f"{'Cluster':<9} | {'Calls':<6} | {'Fail Rate':<10} | {'Avg Drop Turn':<14} | {'Top Topic':<20} | {'Diagnosis':<25}")
        print("-" * 80)

        for c, s in self.cluster_summary.items():
            print(
                f"Cluster {s['cluster_id']:<1} | "
                f"{s['call_count']:<6} | "
                f"{s['failure_rate']:<4}%     | "
                f"Turn {s['avg_drop_turn']:<9} | "
                f"{s['dominant_topic']:<20} | "
                f"{s['diagnosis']:<25}"
            )
        print("=" * 80)


# =====================================================================
# RAG Knowledge Base Auto-Optimizer (Continuous Improvement Loop)
# =====================================================================

class RAGKnowledgeOptimizer:
    """
    Reads failure clusters pinpointed by K-Means and automatically updates
    `Rag_Knowledge_base.txt` with targeted objection scripts and clarifications.
    """

    OPTIMIZATION_PLAYBOOKS = {
        "PRICING_OBJECTION_FRICTION": {
            "title": "Automated Playbook: Flexible Enterprise ROI & Staged Onboarding",
            "category": "objection_handling",
            "content": (
                "When a prospect drops off at pricing or claims high setup costs:\n"
                "1. Offer Staged Rollout: 'We offer a 30-day performance pilot at zero risk, where you only pay if the AI agent hits your qualified lead target.'\n"
                "2. Financial Justification: Emphasize that ApexSales costs less than $1.50 per live discovery conversation vs. $45+ with traditional outsourced SDRs.\n"
                "3. Flexible Monthly Option: Highlight our month-to-month starter tier with no annual commitment."
            )
        },
        "EARLY_GATEKEEPER_BOUNCE": {
            "title": "Automated Playbook: Pattern Interrupt & Executive Hook",
            "category": "objection_handling",
            "content": (
                "When facing immediate 30-second hang-ups or gatekeeper friction:\n"
                "1. High-value Pattern Interrupt: 'I know you weren't expecting my call—I'll be brief. We helped [Competitor/Industry Peer] increase demo bookings by 43% in 3 weeks.'\n"
                "2. Low-friction permission: 'Would you be open to seeing a 60-second summary video before deciding if this is relevant for your team?'"
            )
        },
        "FEATURE_COMPLEXITY_DROP_OFF": {
            "title": "Automated Playbook: Simplified Tech Stack & Zero-Code Setup",
            "category": "objection_handling",
            "content": (
                "When a prospect hesitates on technical complexity or voice latency:\n"
                "1. Reassure Simplicity: 'You don't need an engineering team. ApexSales connects with Salesforce and HubSpot in under 5 minutes with our 1-click OAuth integration.'\n"
                "2. Live Latency Demo: Offer to dial their phone on the spot so they experience the sub-500ms voice speed live."
            )
        }
    }

    def __init__(self, kb_path: str = KB_FILE_PATH):
        self.kb_path = kb_path

    def optimize_knowledge_base(self, cluster_diagnostics: Dict[int, Dict[str, Any]]) -> List[str]:
        """
        Inspects critical/high severity failure clusters and injects new remediation
        scripts into Rag_Knowledge_base.txt if not already present.
        """
        applied_updates = []
        
        if not os.path.exists(self.kb_path):
            logger.warning(f"KB file not found at {self.kb_path}")
            return []

        with open(self.kb_path, "r", encoding="utf-8") as f:
            current_content = f.read()

        new_sections = []
        for c, s in cluster_diagnostics.items():
            diagnosis = s.get("diagnosis")
            if s.get("failure_rate", 0) > 50 and diagnosis in self.OPTIMIZATION_PLAYBOOKS:
                playbook = self.OPTIMIZATION_PLAYBOOKS[diagnosis]
                
                # Check if this playbook title is already in KB
                if playbook["title"] in current_content:
                    logger.info(f"Playbook for {diagnosis} already exists in knowledge base.")
                    continue

                formatted_section = (
                    f"\n[SECTION: {playbook['category'].upper()}]\n"
                    f"Title: {playbook['title']}\n"
                    f"Category: {playbook['category']}\n"
                    f"Content:\n{playbook['content']}\n"
                )
                new_sections.append(formatted_section)
                applied_updates.append(f"Injected '{playbook['title']}' for {diagnosis} (Cluster {c})")

        if new_sections:
            with open(self.kb_path, "a", encoding="utf-8") as f:
                for section in new_sections:
                    f.write(section)
            logger.info(f"Successfully appended {len(new_sections)} optimized playbook sections to {self.kb_path}")

        return applied_updates


# =====================================================================
# Main Execution / Benchmark Demo
# =====================================================================

def run_clustering_and_optimization():
    """Runs full call dataset simulation, K-Means clustering, and auto-optimization."""
    print("=" * 80)
    print("  AI B2B Sales Agent - K-Means Call Clustering & Knowledge Base Optimizer")
    print("=" * 80)

    # 1. Generate or load call dataset
    print("\n[Step 1] Loading and preprocessing 150 recorded sales call transcripts...")
    df_calls = generate_sample_sales_call_logs(num_calls=150)
    print(f"Loaded {len(df_calls)} call logs across features: {SalesCallClustering.FEATURE_COLS}")

    # 2. Fit K-Means
    print("\n[Step 2] Running K-Means clustering algorithm (k=4)...")
    analyzer = SalesCallClustering(k_clusters=4)
    analyzer.fit_and_analyze(df_calls)

    # 3. Print mathematical diagnostics
    analyzer.print_diagnostics()

    # 4. Run automatic RAG optimizer feedback loop
    print("\n[Step 3] Running RAG Knowledge Base Auto-Optimization...")
    optimizer = RAGKnowledgeOptimizer(kb_path=KB_FILE_PATH)
    updates = optimizer.optimize_knowledge_base(analyzer.cluster_summary)

    if updates:
        print("\n✓ Actions Taken to Fix Agent Failures in `Rag_Knowledge_base.txt`:")
        for update in updates:
            print(f"  + {update}")
    else:
        print("\n✓ Knowledge base is already optimized for all identified friction points.")

    print("\n" + "=" * 80)
    print("  Clustering & Continuous Improvement Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_clustering_and_optimization()
