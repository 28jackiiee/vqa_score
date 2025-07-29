#!/bin/bash

# Set up cache directories with fallback if SCRATCH is not defined
if [[ -z "$SCRATCH" ]]; then
    SCRATCH="$(pwd)/scratch"
    echo "[Info] SCRATCH not defined, using fallback: $SCRATCH"
    mkdir -p "$SCRATCH"
fi

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

export TRITON_CACHE_DIR=$SCRATCH/triton_cache

# Ensure cache directories exist and are writable
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

# Simple Parallel GPU Evaluation Orchestrator
# Usage: ./run_parallel_eval.sh input.json ref.json [num_gpus_or_gpu_list]

set -e

# Script arguments
INPUT_FILE="$1"
REF_FILE="$2"
GPU_SPEC="${3:-}"

# Validate arguments
if [[ -z "$INPUT_FILE" ]] || [[ -z "$REF_FILE" ]]; then
    echo "Usage: $0 input.json ref.json [num_gpus_or_gpu_list]"
    echo "  input.json           - Input JSON file with video/label pairs"
    echo "  ref.json             - Reference JSON file with question templates"
    echo "  num_gpus_or_gpu_list - Number of GPUs (e.g., 4) or comma-separated GPU IDs (e.g., 1,2,3,4)"
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

# Parse GPU specification
GPU_LIST=()
if [[ -z "$GPU_SPEC" ]]; then
    # Auto-detect number of GPUs if not specified
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "[Info] Auto-detected $NUM_GPUS GPUs"
        for ((i=0; i<NUM_GPUS; i++)); do
            GPU_LIST+=($i)
        done
    else
        echo "Error: nvidia-smi not found and gpu specification not provided"
        exit 1
    fi
elif [[ "$GPU_SPEC" =~ ^[0-9]+$ ]]; then
    # Numeric specification (number of GPUs starting from 0)
    NUM_GPUS="$GPU_SPEC"
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_LIST+=($i)
    done
elif [[ "$GPU_SPEC" =~ ^[0-9,]+$ ]]; then
    # Comma-separated GPU IDs
    IFS=',' read -ra GPU_LIST <<< "$GPU_SPEC"
    NUM_GPUS=${#GPU_LIST[@]}
else
    echo "Error: Invalid GPU specification '$GPU_SPEC'"
    echo "Use either a number (e.g., 4) or comma-separated GPU IDs (e.g., 1,2,3,4)"
    exit 1
fi

if [[ $NUM_GPUS -eq 0 ]]; then
    echo "Error: No GPUs specified"
    exit 1
fi

echo "[Info] Using ${GPU_LIST[*]} GPUs for parallel evaluation"
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

for ((i=0; i<${#GPU_LIST[@]}; i++)); do
    gpu_id=${GPU_LIST[$i]}
    CHUNK_FILE="$TEMP_DIR/chunk_${i}.json"
    
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
        echo "[Info] No chunk file for GPU index $i (GPU ID $gpu_id), skipping"
    fi
done

# Wait for all GPU processes to complete
echo "[Info] Waiting for all GPU processes to complete..."
echo "[Info] Monitor progress with: tail -f $TEMP_DIR/gpu_*.log"

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    exit_code=$?
    gpu_id=${GPU_LIST[$i]}
    if [[ $exit_code -ne 0 ]]; then
        echo "[Warning] GPU $gpu_id process failed with exit code $exit_code"
        echo "[Warning] Check log: $TEMP_DIR/gpu_${gpu_id}.log"
    else
        echo "[Info] GPU $gpu_id process completed successfully"
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
echo "[Done] Used GPUs: ${GPU_LIST[*]}"