#!/usr/bin/env bash

# Fail fast and be safe in bash
set -euo pipefail

# ========== Visible GPUs (comma-separated) ==========
# Example: GPU_IDS="0,1" uses GPU 0 and 1; GPU_IDS="0" uses a single GPU.
GPU_IDS="0,1,2,3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
# If running in a container, you can also export this (optional):
export NVIDIA_VISIBLE_DEVICES="${GPU_IDS}"

# Project Python path
export PYTHONPATH="/gemini/code/GraphGPT-main:${PYTHONPATH:-}"

# ===== Basic config =====
output_model=path to arxivpub
datapath=path to arxiv_test_instruct_cot.json
graph_data_path=path to all_graph_data.pt
res_path=path to eval_output
start_id=0
end_id=20000 #arxiv
num_gpus=4


# ===== Sweep / naming =====
# SWEEP_PREFIX_BASE is used as the prefix of OUTPUT_TAG.
SWEEP_PREFIX_BASE="arxivpub_arxiv_test_cot_gcd"

# Optional extra args (e.g., cutoff-related). Keep as an array to avoid set -u issues.
CUT_ARG=()

# ===== GCD module params =====
# Whether to enable GCD
ENABLE_GCD=true
# Edge drop rate
GCD_DROP_EDGE_RATE=0.2
# Text contrastive weight
GCD_CD_ALPHA=0.5
# Graph contrastive weight
GCD_CG_BETA=1.0
# Cutoff coefficient
GCD_CUT_PARA=1.0
# Edge count threshold
GCD_EDGE_THRESHOLD=10

# ===== Graph Attention module params =====
# Whether to enable Graph Attention Boost
ENABLE_GRAPH_ATTENTION_BOOST=true
# Attention start layer
GRAPH_ATTN_START_LAYER=15
# Attention end layer
GRAPH_ATTN_END_LAYER=22
# Entropy threshold
GRAPH_ATTN_ENTROPY_THRESH=0.75
# Alpha
GRAPH_ATTN_ALPHA=0.2

# ===== GraphREINJECT module params =====
# Whether to enable GraphREINJECT
ENABLE_GRAPHREINJECT=true
# Reinjection start layer
GRAPH_REINJECT_START_LAYER=8
# Reinjection end layer
GRAPH_REINJECT_END_LAYER=16
# Entropy threshold
GRAPH_REINJECT_ENTROPY_THRESH=0.75
# Retracing ratio
GRAPH_REINJECT_RATIO=0.25


# ===== Output naming (used by run_graphgpt.py to build subfolder / file prefix) =====
OUTPUT_FOLDER_NAME=""
OUTPUT_FILE_PREFIX=""

# Log directory
log_dir=path to logs
mkdir -p "${log_dir}"

# Build optional flags based on boolean switches (argparse store_true: present => true)
GCD_FLAG=()
if [ "${ENABLE_GCD}" = "true" ]; then
  GCD_FLAG+=("--enable-gcd")
fi

ATTN_FLAG=()
if [ "${ENABLE_GRAPH_ATTENTION_BOOST}" = "true" ]; then
  ATTN_FLAG+=("--enable-graph-attention-boost")
fi

GRAPHREINJECT_FLAG=()
if [ "${ENABLE_GRAPHREINJECT}" = "true" ]; then
  GRAPHREINJECT_FLAG+=("--enable-graphreinject")
fi


# Run once with current hyper-parameters (no loop)
OUTPUT_TAG="${SWEEP_PREFIX_BASE}_drop${GCD_DROP_EDGE_RATE}_alpha${GCD_CD_ALPHA}_beta${GCD_CG_BETA}_edge${GCD_EDGE_THRESHOLD}_attn${GRAPH_ATTN_START_LAYER}-${GRAPH_ATTN_END_LAYER}_attnEnt${GRAPH_ATTN_ENTROPY_THRESH}_attnAlpha${GRAPH_ATTN_ALPHA}_reinj${GRAPH_REINJECT_START_LAYER}-${GRAPH_REINJECT_END_LAYER}_reinjEnt${GRAPH_REINJECT_ENTROPY_THRESH}_reinjRatio${GRAPH_REINJECT_RATIO}"
OUTPUT_FOLDER_NAME="${OUTPUT_TAG}"
OUTPUT_FILE_PREFIX="${OUTPUT_TAG}"

run_log="${log_dir}/${OUTPUT_TAG}_$(date +%Y%m%d_%H%M%S).log"

# Start log
    (
      echo "========================================"
      echo "Experiment started at: $(date)"
      echo "Output_model: ${output_model}"
      echo "Res_path:     ${res_path}"
      echo "Data:         ${datapath}"
      echo "Output dir:   ${res_path}"
      echo "Output folder name: ${OUTPUT_FOLDER_NAME}"
      echo "Output file prefix: ${OUTPUT_FILE_PREFIX}"
      echo "GPU IDs:      ${GPU_IDS}"
      echo "GPU(s) ask:   ${num_gpus}"
      echo "---- Module Params ----"
      echo "[GCD] ENABLE_GCD=${ENABLE_GCD:-true} drop_edge_rate=${GCD_DROP_EDGE_RATE} cd_alpha=${GCD_CD_ALPHA} cg_beta=${GCD_CG_BETA} cut_para=${GCD_CUT_PARA} edge_threshold=${GCD_EDGE_THRESHOLD}"
      echo "[GraphAttnBoost] ENABLE_GRAPH_ATTENTION_BOOST=${ENABLE_GRAPH_ATTENTION_BOOST} starting_layer=${GRAPH_ATTN_START_LAYER} ending_layer=${GRAPH_ATTN_END_LAYER} entropy_threshold=${GRAPH_ATTN_ENTROPY_THRESH} alpha=${GRAPH_ATTN_ALPHA}"
      echo "[GraphREINJECT] ENABLE_GRAPHREINJECT=${ENABLE_GRAPHREINJECT} starting_layer=${GRAPH_REINJECT_START_LAYER} ending_layer=${GRAPH_REINJECT_END_LAYER} entropy_threshold=${GRAPH_REINJECT_ENTROPY_THRESH} retracing_ratio=${GRAPH_REINJECT_RATIO}"
      echo "start_id:     ${start_id}"
      echo "end_id:       ${end_id}"
      echo "Log file:     ${run_log}"
      echo "========================================"
    ) | tee "${run_log}"

# Debug: print final switch values before running
    echo "[Debug] Final flag values before check:" | tee -a "${run_log}"
    echo "  ENABLE_GCD=${ENABLE_GCD}" | tee -a "${run_log}"
    echo "  ENABLE_GRAPH_ATTENTION_BOOST=${ENABLE_GRAPH_ATTENTION_BOOST}" | tee -a "${run_log}"
    echo "  ENABLE_GRAPHREINJECT=${ENABLE_GRAPHREINJECT}" | tee -a "${run_log}"
    echo "----" | tee -a "${run_log}"

    set -x
    python3.10 ./graphgpt/eval/run_graphgpt.py \
      --model-name "${output_model}" \
      --prompting_file "${datapath}" \
      --graph_data_path "${graph_data_path}" \
      --output_res_path "${res_path}" \
      --start_id "${start_id}" \
      --end_id "${end_id}" \
      --num_gpus "${num_gpus}" \
      --output-folder-name "${OUTPUT_FOLDER_NAME}" \
      --output-file-prefix "${OUTPUT_FILE_PREFIX}" \
      --gcd-drop-edge-rate "${GCD_DROP_EDGE_RATE}" \
      --gcd-cd-alpha "${GCD_CD_ALPHA}" \
      --gcd-cg-beta "${GCD_CG_BETA}" \
      --gcd-cut-para "${GCD_CUT_PARA}" \
      --gcd-edge-threshold "${GCD_EDGE_THRESHOLD}" \
      --graph-attn-start-layer "${GRAPH_ATTN_START_LAYER}" \
      --graph-attn-end-layer "${GRAPH_ATTN_END_LAYER}" \
      --graph-attn-entropy-thresh "${GRAPH_ATTN_ENTROPY_THRESH}" \
      --graph-attn-alpha "${GRAPH_ATTN_ALPHA}" \
      --graph-reinject-start-layer "${GRAPH_REINJECT_START_LAYER}" \
      --graph-reinject-end-layer "${GRAPH_REINJECT_END_LAYER}" \
      --graph-reinject-entropy-thresh "${GRAPH_REINJECT_ENTROPY_THRESH}" \
      --graph-reinject-retracing-ratio "${GRAPH_REINJECT_RATIO}" \
      "${CUT_ARG[@]}" \
      "${GCD_FLAG[@]}" \
      "${ATTN_FLAG[@]}" \
      "${GRAPHREINJECT_FLAG[@]}" \
      2>&1 | tee -a "${run_log}"
    set +x

# End log
    echo "Finished at: $(date)" | tee -a "${run_log}"
    echo "========================================" | tee -a "${run_log}"

  done
done