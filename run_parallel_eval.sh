#!/bin/bash

# Simple Parallel GPU Evaluation Orchestrator
# Usage: ./run_parallel_eval.sh input.json ref.json [num_gpus]

set -e

# Script arguments
INPUT_FILE="$1"
REF_FILE="$2"
NUM_GPUS="${3:-}"

# Validate arguments
if [[ -z "$INPUT_FILE" ]] || [[ -z "$REF_FILE" ]]; then
    echo "Usage: $0 input.json ref.json [num_gpus]"
    echo "  input.json  - Input JSON file with video/label pairs"
    echo "  ref.json    - Reference JSON file with question templates"
    echo "  num_gpus    - Number of GPUs to use (optional, auto-detects if not provided)"
    exit 1
fi

# Check if files exist
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

if [[ ! -f "$REF_FILE" ]]; then
    echo "Error: Reference file '$REF_FILE' not found"
    exit 1
fi

if [[ ! -f "file_management.py" ]]; then
    echo "Error: file_management.py not found"
    exit 1
fi

if [[ ! -f "score.py" ]]; then
    echo "Error: score.py not found"
    exit 1
fi

# Auto-detect number of GPUs if not specified
if [[ -z "$NUM_GPUS" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "[Info] Auto-detected $NUM_GPUS GPUs"
    else
        echo "Error: nvidia-smi not found and num_gpus not specified"
        exit 1
    fi
fi

if [[ $NUM_GPUS -eq 0 ]]; then
    echo "Error: No GPUs available"
    exit 1
fi

echo "[Info] Using $NUM_GPUS GPUs for parallel evaluation"
echo "[Info] Input file: $INPUT_FILE"
echo "[Info] Reference file: $REF_FILE"

# Get base filename without extension
BASE_NAME=$(basename "$INPUT_FILE" .json)
BASE_DIR=$(dirname "$INPUT_FILE")

# Create temporary directory for chunks
TEMP_DIR="$BASE_DIR/${BASE_NAME}_chunks"
mkdir -p "$TEMP_DIR"

echo "[Info] Creating data chunks in: $TEMP_DIR"

# Split input file into chunks using file_management.py
python3 file_management.py split --input_file "$INPUT_FILE" --num_gpus "$NUM_GPUS" --output_dir "$TEMP_DIR"

# Start background processes for each GPU
echo "[Info] Starting evaluation processes..."
PIDS=()
OUTPUT_FILES=()

for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    CHUNK_FILE="$TEMP_DIR/chunk_${gpu_id}.json"
    
    if [[ -f "$CHUNK_FILE" ]]; then
        LOG_FILE="$TEMP_DIR/gpu_${gpu_id}.log"
        OUTPUT_FILE="${CHUNK_FILE%.json}_scored.json"
        OUTPUT_FILES+=("$OUTPUT_FILE")
        
        echo "[Info] Starting GPU $gpu_id process..."
        
        # Run in background with GPU isolation and output redirect
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            python3 score.py -i "$CHUNK_FILE" -r "$REF_FILE" > "$LOG_FILE" 2>&1
            echo "[GPU $gpu_id] Process completed" >> "$LOG_FILE"
        ) &
        
        PIDS+=($!)
        echo "[Info] GPU $gpu_id process started (PID: ${PIDS[-1]}) - logs: gpu_${gpu_id}.log"
    else
        echo "[Info] No chunk file for GPU $gpu_id, skipping"
    fi
done

# Wait for all GPU processes to complete
echo "[Info] Waiting for all GPU processes to complete..."
echo "[Info] Monitor progress with: tail -f $TEMP_DIR/gpu_*.log"

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "[Warning] GPU $i process failed with exit code $exit_code"
        echo "[Warning] Check log: $TEMP_DIR/gpu_${i}.log"
    else
        echo "[Info] GPU $i process completed successfully"
    fi
done

echo "[Info] All GPU processes completed"

# Merge results using file_management.py
FINAL_OUTPUT="${BASE_DIR}/${BASE_NAME}_scored.json"
echo "[Info] Merging results into: $FINAL_OUTPUT"

python3 file_management.py merge --output_files "${OUTPUT_FILES[@]}" --final_output "$FINAL_OUTPUT"

# Cleanup temporary files
echo "[Info] Cleaning up chunk files..."
rm -f "$TEMP_DIR"/chunk_*.json

echo ""
echo "[Done] Parallel evaluation completed!"
echo "[Done] Results written to: $FINAL_OUTPUT"
echo "[Done] Logs available in: $TEMP_DIR/gpu_*.log"
echo "[Done] Used $NUM_GPUS GPUs"