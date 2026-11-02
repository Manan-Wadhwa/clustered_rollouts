#!/usr/bin/env bash
# Serve a model on an OpenAI-compatible endpoint with vLLM.
#
#   GPUS=1 bash serve_vllm.sh                      # Qwen3-8B, one GPU, thinking off
#   GPUS=4 MODE=dp bash serve_vllm.sh              # 4 replicas behind one port (best for an 8B model)
#   GPUS=2 MODE=tp MODEL=<judge> bash serve_vllm.sh   # shard a larger judge model
#   THINKING=on bash serve_vllm.sh                 # Qwen3 thinking mode, <think> split out by the parser
#
# The served model name defaults to MODEL, so the suite config must list exactly
# the same string ("Qwen/Qwen3-8B").
#
# Thinking. Qwen3 thinks by default. The suite's dev system prompt already gives
# the model a <SCRATCHPAD_REASONING> block, so thinking mode adds a second
# reasoning channel. The flag on this server only decides whether vLLM splits
# <think> out of the reply. Whether the model thinks at all is set per request
# by the client (OPENROUTER_EXTRA_BODY in the suite's .env, see PATCH_NOTES.md):
#   THINKING=off: no reasoning parser, and the client sends enable_thinking=false.
#                 raw_response holds the whole reply, scratchpad included.
#   THINKING=on:  --reasoning-parser qwen3. vLLM moves <think>...</think> into
#                 reasoning_content, which the suite never saves. raw_response
#                 holds only the final answer, so the judge never sees drafts
#                 written while thinking. Leave OPENROUTER_EXTRA_BODY unset.
#                 Without the parser, <think> text would reach raw_response, and
#                 an email drafted inside <think> could pass the classifiers'
#                 tool_use:email checks.
#
# Sampling. The suite sends temperature=1.0 and nothing else. By default vLLM
# fills top_p/top_k from the model's generation_config.json (Qwen3-8B: top_p 0.95,
# top_k 20). GENCFG=vllm (the default here) turns that off, so sampling is plain
# temperature 1.0 as the config states. GENCFG=auto keeps Qwen's defaults.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-8000}
GPUS=${GPUS:-1}
MODE=${MODE:-dp}            # dp: one full copy per GPU; tp: shard one copy across GPUs
THINKING=${THINKING:-off}
MAX_LEN=${MAX_LEN:-16384}   # longest prompt is ~2.7k tokens; the suite asks for max_tokens=10000
MAX_SEQS=${MAX_SEQS:-256}   # concurrent sequences per replica
GENCFG=${GENCFG:-vllm}
VLLM=${VLLM:-vllm}

args=(serve "$MODEL"
  --host 127.0.0.1 --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization 0.90
  --enable-prefix-caching     # 50-100 samples per condition share one prompt
  --generation-config "$GENCFG"
)

if [[ "$GPUS" -gt 1 ]]; then
  if [[ "$MODE" == tp ]]; then
    args+=(--tensor-parallel-size "$GPUS")
  else
    args+=(--data-parallel-size "$GPUS")
  fi
fi

# Parser name is "qwen3" in recent vLLM; older releases used "deepseek_r1".
# Check with: vllm serve --help=all | grep -A2 reasoning-parser  (plain --help on older releases)
[[ "$THINKING" == on ]] && args+=(--reasoning-parser qwen3)

echo "+ $VLLM ${args[*]}" >&2
exec "$VLLM" "${args[@]}"
