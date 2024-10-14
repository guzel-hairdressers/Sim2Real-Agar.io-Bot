#!/usr/bin/env bash
# ==============================================================================
# Execution Script: Train PPO Navigation Policy & Run Sim2Real Robustness Benchmark
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/../../.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "    PPO SIM2REAL CONTINUOUS NAVIGATION - TRAINING & EVALUATION       "
echo "======================================================================"

# Step 1: Run Training / Rollout Simulation
echo "[1/3] Training PPO Navigation Policy..."
"${PYTHON_BIN}" src/train.py --timesteps 50000 --output outputs/ppo_navigation_policy.zip

# Step 2: Run Sim2Real Robustness Evaluation Benchmark
echo "[2/3] Evaluating Sim2Real Parameter Shift Robustness..."
"${PYTHON_BIN}" src/evaluate_robustness.py --out outputs/robustness_results.csv

# Step 3: Export Policy to ONNX for Embedded Edge Deployment
echo "[3/3] Exporting Policy to ONNX..."
"${PYTHON_BIN}" src/export_onnx.py --out outputs/navigation_policy.onnx

echo "======================================================================"
echo " Pipeline Complete! Artifacts Generated in outputs/ directory:"
ls -lh outputs/
echo "======================================================================"
