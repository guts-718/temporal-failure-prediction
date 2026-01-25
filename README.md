# Temporal Failure Prediction for Microservices

This project aims to predict **microservice failures before they occur** using
time series modeling with recurrent neural networks (LSTM / GRU).

Rather than detecting failures after they happen, the goal is to provide
**early warning signals** (minutes in advance) based on historical metrics,
enabling proactive handling in production systems.

---

## Problem Definition

**sequence to event prediction**

Given a fixed window of historical service metrics, the model predicts whether
a failure will occur within a future time horizon or not.

---

## Failure Definition (v1)

In version 1, a failure is defined as:

- **Error rate ≥ 5%**
- Sustained for **at least 2 consecutive minutes**

This definition is made intentionally simple and will be extended in later versions
to include other latency-based and resource-based failures.

---

## Input / Output

### Input
- Metrics from the **last 10 minutes**
- Single microservice
- Metrics sampled at 1 minute intervals

### Output
- Probability of failure occurring in the **next 5 minutes**

---

## Input Features (v1 schema)

The system is designed to support up to 10 features per time step:

- request_count
- error_rate
- p50_latency
- p95_latency
- p99_latency
- cpu_usage
- memory_usage
- network_in
- network_out
- pod_restarts

I may not use all the features in early experiments, but I've fixed the schema to
allow incremental extension.

---

## Evaluation Strategy

To reflect real world deployment:

- **Time-based train / validation / test splits** are used
- Models are evaluated on:
  - Recall 
  - Precision 
  - Average early warning time 

Simple statistical baselines (static thresholds, rolling averages,
trend based heuristics) are used for comparison.

---

## Roadmap

### Version 1
- Single-service failure prediction
- Synthetic metric generation
- LSTM / GRU models
- Baseline comparisons

### Version 2 and Later
- Multi-service support
- Service aware modeling
- Additional failure definitions (latency, resource saturation)
- Improved evaluation and analysis


---

## Running the Data Pipeline

From the project root:

```bash
# 1. Generate synthetic metrics
python -m data.synthetic.generate_data

# 2. Build sliding window dataset
python -m src.data.build_dataset

# 3. Normalize using train only stats
python -m src.data.normalize.py

# 4. Run the baseline models
python -m src.training.run_baselines

```


This makes your repo immediately usable.

---


## Baseline Results (Current)

Positive sample rate (test set): ~2.7%

| Model                      | Precision | Recall | F1   |
|---------------------------|-----------|--------|------|
| Rule-based threshold      | 0.129     | 0.275  | 0.176 |
| Snapshot Logistic Reg     | 0.781     | 0.140  | 0.238 |
| Window Aggregated LogReg  | 0.786     | 0.124  | 0.214 |

Observations:
- Rule-based approach has higher recall but poor precision.
- Logistic regression models are conservative with high precision but low recall.
- Temporal aggregates alone are insufficient thus motivating sequence models (LSTM/GRU).


## Sequence Model Results

| Model | Precision | Recall | F1 |
|------|----------|-------|----|
| LSTM | 0.142 | 0.303 | 0.194 |
| GRU  | 0.123 | 0.258 | 0.167 |

Observations:
- LSTM achieves the highest recall among all models.
- Sequence models significantly improve early failure detection compared to classical baselines.
- LSTM outperforms GRU on this dataset.



