#!/bin/bash

# ChatGPT Model 6 and Claude Sonnet 5.5 were used to assist with code development, clean-up, and documentation

# launcher for simulation script
#
# RECOMMENDED USE:
#     bash run_python_slurm_simulation.sh
#
# When launched from the command line, this script reads SENSORS and
# N_RUNS_PER_SENSOR from the Python simulation file, calculates the required
# Slurm array size automatically, and submits itself with sbatch.
#
# Once inside a Slurm array task, it activates the PyMC environment and runs
# exactly one (sensor, run) simulation.

#SBATCH --job-name=reset_sim
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reset_sim_%A_%a.out
#SBATCH --error=logs/reset_sim_%A_%a.err

set -euo pipefail

# User-adjustable launcher settings
PYTHON_SCRIPT="biosensor_simulation_general.py"
VENV_PATH="$HOME/pymc_env"
MAX_SIMULTANEOUS_TASKS=20

# If not already in Slurm, determine the array size and submit
# if [[ -z "${SLURM_JOB_ID:-}" ]]; then
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p logs

    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo "ERROR: Cannot find $PYTHON_SCRIPT in $(pwd)"
        echo "Run this launcher from the directory containing the simulation script."
        exit 1
    fi

    # Use the same Python environment that will run the simulation.
    source "$VENV_PATH/bin/activate"

    N_TOTAL=$(python - <<PY
import importlib.util
from pathlib import Path

path = Path("$PYTHON_SCRIPT")
spec = importlib.util.spec_from_file_location("reset_sim_config", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

n_sensors = len(mod.SENSORS)
n_runs = int(mod.N_RUNS_PER_SENSOR)
print(n_sensors * n_runs)
PY
)

    if [[ "$N_TOTAL" -lt 1 ]]; then
        echo "ERROR: Calculated zero Slurm tasks."
        exit 1
    fi

    LAST_TASK=$((N_TOTAL - 1))

    echo "======================================"
    echo "Submitting resetting simulation"
    echo "Python script:      $PYTHON_SCRIPT"
    echo "Total array tasks:  $N_TOTAL"
    echo "Array IDs:          0-$LAST_TASK"
    echo "Max simultaneous:   $MAX_SIMULTANEOUS_TASKS"
    echo "======================================"

    sbatch \
        --array="0-${LAST_TASK}%${MAX_SIMULTANEOUS_TASKS}" \
        "$0"

    exit 0
fi

# ------------------------------------------------------------
# We are now inside one Slurm array task
# ------------------------------------------------------------
mkdir -p logs
source "$VENV_PATH/bin/activate"

echo "======================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node:          ${SLURMD_NODENAME:-NA}"
echo "Python:        $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "Working dir:   $(pwd)"
echo "======================================"

python "$PYTHON_SCRIPT"
