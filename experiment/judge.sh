#!/usr/bin/env bash
# Box-side: after every generation job on this box has printed JOB_DONE, stop the
# generation servers, serve gpt-oss-120b as the judge, and run the suite's own
# classify_results.py on every results directory (same prompts and email context
# as the original Claude judge). Labels land in each response.json "classification".
set -euo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
G=/marimo/g022
JUDGE=openai/gpt-oss-120b
PORT=8100
exec > >(tee -a $G/logs/judge.log) 2>&1
echo "== judge $(date -u)"

jobs=$(cut -d' ' -f1 $G/jobs_started)
for j in $jobs; do
  # Last line only: driver logs are appended across restarts and keep old JOB_DONE lines.
  until tail -n 1 $G/logs/$j.driver.log 2>/dev/null | grep -q "^JOB_DONE $j"; do sleep 30; done
  echo "generation done: $j"
done
pkill -9 -f "vllm serve" || true
pkill -9 -f EngineCore || true
sleep 10

VLLM_BIN=$(ls $G/venv/bin/vllm 2>/dev/null || command -v vllm || echo /tmp/uv-venv/bin/vllm)
nohup $VLLM_BIN serve $JUDGE --host 127.0.0.1 --port $PORT --max-model-len 16384 \
  --max-num-seqs 256 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --attention-backend TRITON_ATTN > $G/logs/judge.vllm.log 2>&1 &
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "judge server did not come up"; tail -40 $G/logs/judge.vllm.log; exit 1; }
echo "judge up"

unset PYTHONPATH OPENROUTER_EXTRA_BODY
export OPENROUTER_API_KEY=EMPTY OPENROUTER_BASE_URL=http://127.0.0.1:$PORT/v1 AM_FORCE_OPENROUTER=1
cd $G/suite
for j in $jobs; do
  for pass in 1 2 3; do  # rerun: files whose judge reply could not be parsed stay unclassified
    $G/suitevenv/bin/python scripts/classify_results.py --results-dir results/$j \
      --classifier-model $JUDGE --concurrency 128 > $G/logs/judge.$j.$pass.log 2>&1 || true
    tail -2 $G/logs/judge.$j.$pass.log
    grep -q "already classified" $G/logs/judge.$j.$pass.log && break
  done
done
echo "JUDGE_DONE $(date -u)"
