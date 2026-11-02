#!/usr/bin/env bash
# Box-side: wait for g022 setup, patch the suite, build its venv, then start jobs.
#   stage_and_launch.sh "<job> <model> <port> <mem> <thinking> [vllm args]" ["<job2> ..."]
# Each quoted argument is one run_job.sh invocation. Jobs start 240 s apart so two
# vLLM servers on one GPU do not profile memory at the same moment.
set -euo pipefail
G=/marimo/g022
mkdir -p $G/logs
exec > >(tee -a $G/logs/stage.log) 2>&1
echo "== stage $(date -u)"

until grep -q SETUP_DONE $G/setup.log 2>/dev/null; do sleep 20; done
echo "setup done"

cd $G/suite
echo "suite HEAD: $(git rev-parse HEAD)"
# The same three edits as patches/local_vllm.diff, plus AM_FORCE_OPENROUTER, done as
# exact string replacements that must each match once.
python3 - <<'EOF'
import pathlib
mc = pathlib.Path("api_client/model_client.py")
s = mc.read_text()
if "OPENROUTER_BASE_URL" not in s:
    old = 'base_url="https://openrouter.ai/api/v1"'
    assert s.count(old) == 1, "base_url"
    s = s.replace(old, 'base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")')
    old = "        api_start = time.time()\n        response = await self.openrouter_client.chat.completions.create("
    assert s.count(old) == 1, "openrouter call"
    s = s.replace(old, (
        "        extra_body = os.getenv(\"OPENROUTER_EXTRA_BODY\")\n"
        "        if extra_body and \"extra_body\" not in kwargs:\n"
        "            import json as _json\n"
        "            kwargs[\"extra_body\"] = _json.loads(extra_body)\n") + old)
    old = '        """Detect the appropriate provider for a given model ID."""\n'
    assert s.count(old) == 1, "detect_provider"
    # Route any "org/model" id to the local server; otherwise "openai/gpt-oss-20b"
    # would go to the OpenAI client because its name contains "gpt".
    s = s.replace(old, old + '        if os.getenv("AM_FORCE_OPENROUTER") and "/" in model_id:\n            return "openrouter"\n')
    mc.write_text(s)
re_ = pathlib.Path("scripts/run_experiments.py")
s = re_.read_text()
if "AM_MAX_MODEL_FAILURES" not in s:
    old = "MAX_MODEL_FAILURES = 3"
    assert s.count(old) == 1, "max failures"
    s = s.replace(old, 'MAX_MODEL_FAILURES = int(__import__("os").getenv("AM_MAX_MODEL_FAILURES", "3"))')
    re_.write_text(s)
print("suite patched")
EOF
git diff --stat

if [[ ! -x $G/suitevenv/bin/python ]]; then
  uv venv $G/suitevenv -p 3.12 --seed
  # --python, not VIRTUAL_ENV: on molab uv otherwise installs into the kernel environment.
  uv pip install --python $G/suitevenv/bin/python -r requirements.txt
fi
AM_FORCE_OPENROUTER=1 OPENROUTER_API_KEY=EMPTY $G/suitevenv/bin/python -c "
from api_client.model_client import ModelClient
c = ModelClient()
print('route check:', {m: c._detect_provider(m) for m in ['openai/gpt-oss-20b', 'Qwen/Qwen3-8B', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506']})" || echo "route check failed (non-fatal)"

first=1
for spec in "$@"; do
  [[ $first == 1 ]] || sleep 240
  first=0
  # shellcheck disable=SC2086
  nohup bash $G/cr/experiment/run_job.sh $spec > /dev/null 2>&1 &
  echo "started: $spec (pid $!)"
done
echo "== all jobs started $(date -u)"
