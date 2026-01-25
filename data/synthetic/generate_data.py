import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Configs
SEED = 42
TOTAL_DAYS = 30
SAMPLING_MINUTES = 1

FAILURE_ERROR_THRESHOLD = 0.05
FAILURE_DURATION_MINUTES = 2

# Target =  approx 1 failure every 3–8 hours
STRESS_INCREASE_PROB = 0.03
STRESS_DECAY_PROB = 0.05
STRESS_SPIKE_PROB = 0.006
RECOVERY_DECAY_RATE = 0.15

RAW_DATA_DIR = Path("./data/synthetic")
OUT_DIR = Path("./data/processed")
np.random.seed(SEED)

# Helper
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

# simulation
minutes = TOTAL_DAYS * 24 * 60
timestamps = pd.date_range(
    start="2024-01-01",
    periods=minutes,
    freq=f"{SAMPLING_MINUTES}min"
)

stress = 0.0
failure_active = 0
failure_counter = 0

metrics = []
failures = []

consecutive_failure_minutes = 0

for ts in timestamps:
    if np.random.rand() < STRESS_INCREASE_PROB:
        stress += np.random.uniform(0.01, 0.04)

    if np.random.rand() < STRESS_DECAY_PROB:
        stress -= np.random.uniform(0.01, 0.03)

    if np.random.rand() < STRESS_SPIKE_PROB:
        stress += np.random.uniform(0.15, 0.3)

    if failure_active:
        stress -= RECOVERY_DECAY_RATE

    stress = clamp(stress)

    # Base metrics
    request_count = int(np.random.normal(1200, 150))
    cpu_usage = clamp(0.3 + 0.6 * stress + np.random.normal(0, 0.03))
    memory_usage = clamp(0.4 + 0.4 * stress + np.random.normal(0, 0.03))

    p50_latency = 100 + 120 * stress + np.random.normal(0, 10)
    p95_latency = 160 + 220 * stress + np.random.normal(0, 20)
    p99_latency = 220 + 350 * stress + np.random.normal(0, 30)

    error_rate = clamp(
        0.002 + (stress ** 3) * 0.2 + np.random.normal(0, 0.002),
        0.0,
        0.5
    )

    network_in = 200 + 300 * stress + np.random.normal(0, 20)
    network_out = 180 + 280 * stress + np.random.normal(0, 20)

    pod_restarts = 1 if stress > 0.85 and np.random.rand() < 0.05 else 0

    # detecting failure
    if error_rate >= FAILURE_ERROR_THRESHOLD:
        consecutive_failure_minutes += 1
    else:
        consecutive_failure_minutes = 0

    if consecutive_failure_minutes >= FAILURE_DURATION_MINUTES:
        failure_active = 1
    else:
        failure_active = 0

    metrics.append([
        ts, request_count, error_rate,
        p50_latency, p95_latency, p99_latency,
        cpu_usage, memory_usage,
        network_in, network_out,
        pod_restarts
    ])

    failures.append([ts, failure_active])

metrics_df = pd.DataFrame(
    metrics,
    columns=[
        "timestamp", "request_count", "error_rate",
        "p50_latency", "p95_latency", "p99_latency",
        "cpu_usage", "memory_usage",
        "network_in", "network_out",
        "pod_restarts"
    ]
)

failures_df = pd.DataFrame(
    failures,
    columns=["timestamp", "failure_active"]
)

metrics_df.to_csv(RAW_DATA_DIR/"metrics.csv", index=False)
failures_df.to_csv(RAW_DATA_DIR/"failures.csv", index=False)

print("Synthetic data generated.")
print(f"Total minutes: {minutes}")
print(f"Total failure minutes: {failures_df['failure_active'].sum()}")
