#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_training_per_model.sh [TRAINING_ARGS...] [--models MODEL1 MODEL2]
# Example: ./run_training_per_model.sh --data credit_card_transactions.csv --target is_fraud --models CatBoost XGBoost

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_SCRIPT="$SCRIPT_DIR/training.py"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_DIR="$REPO_ROOT/experiment-results"
mkdir -p "$EXP_DIR"

# Default models to run
MODELS=("CatBoost" "XGBoost" "LightGBM")

# Parse arguments - separate training args from model selection
TRAINING_ARGS=()
MODEL_ARGS=()
IN_MODELS_SECTION=false

for arg in "$@"; do
    if [[ "$arg" == "--models" ]]; then
        IN_MODELS_SECTION=true
        MODELS=()
    elif [[ "$IN_MODELS_SECTION" == true ]]; then
        if [[ "$arg" == --* ]]; then
            # Next argument starts with --, so we're done with models
            IN_MODELS_SECTION=false
            TRAINING_ARGS+=("$arg")
        else
            MODELS+=("$arg")
        fi
    else
        TRAINING_ARGS+=("$arg")
    fi
done

echo "Models to run: ${MODELS[*]}"
echo "Training args: ${TRAINING_ARGS[*]}"

# Extract dataset name from arguments for better naming
DATASET_NAME="unknown"
for i in "${!TRAINING_ARGS[@]}"; do
    if [[ "${TRAINING_ARGS[i]}" == "--data" ]]; then
        data_arg="${TRAINING_ARGS[i+1]}"
        DATASET_NAME=$(basename "$data_arg" | sed 's/\.[^.]*$//')  # Remove extension
        break
    fi
done

TS="$(date +%Y%m%d-%H%M%S)"
ENERGY_CSV="$EXP_DIR/model_energy.csv"

# Initialize energy CSV
if [[ ! -f "$ENERGY_CSV" ]]; then
    echo "timestamp,dataset,model,energy_joules,execution_seconds,mean_power_w,training_duration_s,exit_code" >> "$ENERGY_CSV"
fi

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
  echo "ERROR: Training script not found: $TRAINING_SCRIPT" >&2
  exit 1
fi

# Function to run a single model with energy measurement
run_model_with_energy() {
    local model="$1"
    local training_args=("${@:2}")

    local model_ts="$(date +%Y%m%d-%H%M%S)"
    local model_prefix="$EXP_DIR/${model_ts}_${DATASET_NAME}_${model}"
    local energibridge_out="${model_prefix}_energibridge.txt"
    local energibridge_pid=""

    echo ""
    echo "=== STARTING MODEL: $model ==="
    echo "Training command: python $TRAINING_SCRIPT ${training_args[*]}"

    # Start energibridge if available
    if command -v energibridge >/dev/null 2>&1; then
        echo "Starting energibridge for $model..."
        energibridge --summary -i 100 -o "${model_prefix}_energibridge.csv" sleep 1e9 >"$energibridge_out" 2>&1 &
        energibridge_pid=$!
        echo "energibridge started with PID $energibridge_pid"
        # Give energibridge a moment to start
        sleep 2
    else
        echo "WARN: energibridge not found; skipping energy measurement for $model" >&2
    fi

    # Run training for specific model by setting environment variable
    export LOAD_MODEL="$model"  # This can be used if your script supports model selection

    local start_time=$(date +%s.%N)
    python "$TRAINING_SCRIPT" "${training_args[@]}"
    local exit_code=$?
    local end_time=$(date +%s.%N)

    local training_duration=$(echo "$end_time - $start_time" | bc)

    # Stop energibridge if it was started
    if [[ -n "$energibridge_pid" ]]; then
        kill "$energibridge_pid" 2>/dev/null || true
        wait "$energibridge_pid" 2>/dev/null || true

        # Extract energy results
        local energy_joules=""
        local execution_seconds=""
        local mean_power=""

        if [[ -f "$energibridge_out" ]]; then
            energy_joules=$(grep "Energy consumption in joules:" "$energibridge_out" | awk '{print $5}' | head -1 || echo "")
            execution_seconds=$(grep "Energy consumption in joules:" "$energibridge_out" | awk '{print $7}' | head -1 || echo "")

            if [[ -n "$energy_joules" && -n "$execution_seconds" ]]; then
                mean_power=$(echo "$energy_joules / $execution_seconds" | bc -l 2>/dev/null || echo "")
            fi
        fi

        # Append to energy CSV
        echo "$model_ts,$DATASET_NAME,$model,$energy_joules,$execution_seconds,$mean_power,$training_duration,$exit_code" >> "$ENERGY_CSV"

        echo "=== $model COMPLETED ==="
        echo "Exit code: $exit_code"
        echo "Training duration: ${training_duration}s"
        echo "Energy consumed: ${energy_joules} J"
        echo "Execution time: ${execution_seconds} s"
        echo "Mean power: ${mean_power} W"
    else
        # No energy measurement, just record timing
        echo "$model_ts,$DATASET_NAME,$model,,,$training_duration,$exit_code" >> "$ENERGY_CSV"

        echo "=== $model COMPLETED ==="
        echo "Exit code: $exit_code"
        echo "Training duration: ${training_duration}s"
        echo "No energy measurement available"
    fi

    # Save individual model summary
    local summary_file="${model_prefix}_summary.json"
    cat > "$summary_file" << EOF
{
  "timestamp": "$model_ts",
  "dataset": "$DATASET_NAME",
  "model": "$model",
  "training_duration_seconds": $training_duration,
  "exit_code": $exit_code,
  "energy_joules": "$energy_joules",
  "execution_seconds": "$execution_seconds",
  "mean_power_watts": "$mean_power",
  "command": "python $TRAINING_SCRIPT ${training_args[*]}"
}
EOF

    echo "Model summary saved to: $summary_file"
    echo ""

    return $exit_code
}

# Function to run using model-specific training (if your script supports it)
run_model_specific_training() {
    local model="$1"
    local training_args=("${@:2}")

    # Create a temporary script for this specific model run
    local temp_script=$(mktemp)
    cat > "$temp_script" << 'EOF'
import sys
import os
from training import main as training_main

# Set model environment if needed
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    # You can modify training behavior based on model here
    # For example, set environment variables or modify args
    os.environ['CURRENT_MODEL'] = model_name

# Run the training with remaining arguments
sys.argv = ['training'] + sys.argv[2:]
training_main()
EOF

    # Run the model-specific training
    python "$temp_script" "$model" "${training_args[@]}"
    local exit_code=$?

    # Clean up
    rm -f "$temp_script"
    return $exit_code
}

# Main execution
echo "Starting per-model training benchmark"
echo "Dataset: $DATASET_NAME"
echo "Total models: ${#MODELS[@]}"
echo "Energy results will be saved to: $ENERGY_CSV"
echo ""

OVERALL_EXIT_CODE=0
for model in "${MODELS[@]}"; do
    if run_model_with_energy "$model" "${TRAINING_ARGS[@]}"; then
        echo "✓ $model completed successfully"
    else
        echo "✗ $model failed with exit code: $?"
        OVERALL_EXIT_CODE=1
    fi
    # Add a small delay between model runs
    sleep 5
done

echo ""
echo "=== ALL MODELS COMPLETED ==="
echo "Energy results saved to: $ENERGY_CSV"
echo "Overall exit code: $OVERALL_EXIT_CODE"

# Print summary table
if [[ -f "$ENERGY_CSV" ]]; then
    echo ""
    echo "=== ENERGY CONSUMPTION SUMMARY ==="
    tail -n +2 "$ENERGY_CSV" | while IFS=, read -r timestamp dataset model energy_joules execution_seconds mean_power training_duration exit_code; do
        printf "%-12s | Energy: %8s J | Time: %6s s | Power: %6s W | Status: %s\n" \
               "$model" "${energy_joules:-N/A}" "${training_duration:-N/A}" "${mean_power:-N/A}" "$([ "$exit_code" -eq 0 ] && echo '✓' || echo '✗')"
    done
fi

exit $OVERALL_EXIT_CODE
