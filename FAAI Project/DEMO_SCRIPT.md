# ⏱️ 2-Minute Faculty Demonstration Script: BreatheOps AI

**Project**: BreatheOps AI — Multi-Agent Air Quality Event Investigation and Advisory System  
**Presenter**: Student Demonstration  
**Target Duration**: ~2 Minutes  

---

## 🎯 Demonstration Objective
Demonstrate how BreatheOps AI replaces static, error-prone threshold alerts with **explainable statistical baselines (Median + MAD)** and an **autonomous multi-agent investigation workflow** that evaluates spatial context, rules out sensor anomalies, enforces scientific safety guardrails, and compiles a validated advisory dossier.

---

## 🕒 Step-by-Step Presentation Walkthrough

### 1. The Problem & Introduction (0:00 – 0:20)
> *"Good morning, Professors. Most air quality monitoring systems rely on static thresholds or black-box models that trigger false alarms and provide zero operational context. **BreatheOps AI** is an explainable multi-agent investigation system. Instead of simply flagging a number, it launches a coordinated team of six specialized logic-based agents to determine whether an anomaly is real, whether it is localized or regional, what weather conditions coincided, and what specific technical actions are required."*

---

### 2. Live Execution Setup (0:20 – 0:40)
> *"The system is completely self-contained, working 100% offline without external APIs or paid services."*

* **Action 1**: Look at the left sidebar under **1. Data Source Selection**.
  - Leave **`Built-in Demo Data (Synthetic)`** selected.
  - *Point out*: Notice the prominent warning badge in the UI: `⚠️ SYNTHETIC DEMONSTRATION DATA (Fixed Seed 42)` — ensuring complete transparency.
* **Action 2**: In the sidebar under **2. Target Station & Pollutant**:
  - Keep Station: **`Aotizhongxin`**
  - Keep Target Pollutant: **`PM2.5`**
* **Action 3**: Click the blue primary button: **`🚀 Run Investigation`**.
  - *Point out*: *"In under one second, the orchestrator coordinates all six agents sequentially: Planning, Data Quality, Detection, Spatial Context, Response Planning, and Critic Certification."*

---

### 3. Overview & Empirical Evidence (0:40 – 1:15)
* **Look at Section 1 (Investigation Overview & Key Results)**:
  - *Point to the Metric Cards*:
    - **Station Baseline (Median)**: `58.0 µg/m³`
    - **Anomaly Threshold**: `109.8 µg/m³` (derived from $1.4826 \times \text{MAD}$)
    - **Anomalies Detected**: `10 anomalous readings`
    - **Spatial Context**: `📍 Isolated Local`
* **Look at Section 2 (Pollution Trends)**:
  - *Point to the Interactive Plotly Chart*:
    - *"Here we see the temporal profile. The green dotted line is the station's historical median; the red dashed line is the robust threshold. Hovering reveals the 10-hour spike peaking at nearly 400 µg/m³ (6.8× the baseline)."*

---

### 4. Cross-Station & Multi-Agent Context (1:15 – 1:40)
* **Look at Section 3 (Data-Quality & Cross-Station Evidence)**:
  - *Explain*: *"The **Multi-Station Context Agent** compared concurrent readings from Changping, Dongsi, and Wanliu during this exact spike window. Those stations remained nominal, proving this event was isolated to Aotizhongxin. Concurrently, meteorological sensors recorded stagnant wind speeds under 0.5 m/s, explaining why local emissions could not disperse."*
* **Look at Section 4 & 5 (Execution Log & Critic Certification)**:
  - *Point to the `🛡️ VALIDATED` badge*:
    - *"Crucially, the **Critic & Safety Agent** acts as an internal auditor. It mathematically cross-checked that the reported counts match the data, eliminated any unsupported causal claims (e.g. blaming a factory without proof), and verified that non-regulatory disclaimers are in place."*

---

### 5. Actionable Advisory & Report Export (1:40 – 2:00)
* **Look at Section 5 (Recommended Actions)**:
  - *Explain*: *"Because the spike was isolated to a single station, the **Response Planning Agent** did NOT declare a false citywide emergency. Instead, it recommended practical field actions: inspecting station Aotizhongxin's optical inlet lines for obstruction, calibrating the sensor, and conducting boundary layer meteorological review."*
* **Look at Section 6 (Final Investigation Report)**:
  - *Action*: Click **`📥 Download Report (.md)`** or **`📄 Download Report (.txt)`**.
  - *Closing*: *"With one click, field operators can download a complete, certified investigation dossier. In Phase 3, BreatheOps AI delivers a dependable, reproducible, and explainable foundation for environmental monitoring."*

---

## 💡 Anticipated Faculty Questions & Model Answers

### Q1: *"Why use Median and MAD instead of standard Mean and Standard Deviation?"*
> **Answer**: *"Environmental sensor data often contains extreme outliers and skewed distributions. The mean and standard deviation are heavily distorted by the very spikes we are trying to detect. The Median and Median Absolute Deviation (scaled by 1.4826 for normal consistency) are 50% breakdown-point robust estimators, meaning historical baselines remain stable even during extreme pollution events."*

### Q2: *"Why are the agents implemented with deterministic Python logic rather than LLMs?"*
> **Answer**: *"For critical environmental and public safety monitoring, hallucinations, API downtime, high token costs, and non-deterministic outputs are major liabilities. By using deterministic, state-passing agents with rigorous tool execution, we guarantee 100% mathematical reproducibility, zero external dependencies, and complete explainability for regulatory audits."*

### Q3: *"How does the system distinguish real from synthetic data?"*
> **Answer**: *"Every dataset is tagged upon ingestion with metadata attributes and visual UI banners. The built-in demo uses a fixed random seed (seed=42) with an explicitly engineered spike for reproducible faculty demonstration, while uploaded files undergo full schema validation without modifying original records."*
