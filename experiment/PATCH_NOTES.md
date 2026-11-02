# Running the suite against a local vLLM server

Suite version read for these notes: `anthropic-experimental/agentic-misalignment` at commit `ea0630e1a3eaae7f9f9740fd2703229d3854ccda` (2025-06-19). Pin it with `git checkout` on the GPU box, since the patch line numbers come from this commit.

## How the suite picks a provider

`api_client/model_client.py` maps a model string to a provider in two places, `get_provider_for_model()` (used by `run_experiments.py` for concurrency limits) and `ModelClient._detect_provider()` (used for the call). Both first check fixed model sets, then fall back to string rules. In `get_provider_for_model`, any id containing `/` goes to `openrouter`. In `_detect_provider`, any id containing `qwen`, `llama` or `deepseek` goes to `openrouter`. So `Qwen/Qwen3-8B` lands on `openrouter` in both, with no code change.

The `openrouter` client is an `openai.AsyncOpenAI` built only when `OPENROUTER_API_KEY` is set, with `base_url="https://openrouter.ai/api/v1"` hard-coded. Because `base_url` is passed explicitly, the `OPENAI_BASE_URL` environment variable has no effect. So the stock suite cannot reach a local server, and a patch is needed.

## The patch (`patches/local_vllm.diff`)

Three changes, 9 added lines, applied with `git apply`:

1. `model_client.py`, `_setup_clients`: `base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")`. This is the only change strictly required.
2. `model_client.py`, `_call_openrouter`: if `OPENROUTER_EXTRA_BODY` is set, parse it as JSON and send it as `extra_body`. This is how the client sends `{"chat_template_kwargs": {"enable_thinking": false}}` to turn Qwen3 thinking off per request. It applies to every `openrouter` call, judge calls included.
3. `run_experiments.py`: `MAX_MODEL_FAILURES = int(os.getenv("AM_MAX_MODEL_FAILURES", "3"))`. The stock runner counts failures per model over the whole run (timeouts, exceptions and empty replies all count) and, after 3, marks the model "isolated" and skips every remaining task for it. With one model and thousands of calls, three bad replies would end the run. Set it to a large number.

Tested on CPU against a stub OpenAI-compatible server (`generate_prompts` → `run_experiments` → `classify_results`, 2 conditions x 5 samples). The stub logged `"model": "Qwen/Qwen3-8B", "max_tokens": 10000, "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": false}` on every subject request, and `classify_results.py --classifier-model Qwen/Qwen3-32B` sent judge calls to the same base URL.

## Environment (the suite's `.env`, read from the working directory)

```
OPENROUTER_API_KEY=EMPTY
OPENROUTER_BASE_URL=http://127.0.0.1:8000/v1
OPENROUTER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
AM_MAX_MODEL_FAILURES=1000000
# ANTHROPIC_API_KEY=...   only for classifier option (a)
```

`OPENROUTER_API_KEY` must be non-empty or the client is never built. vLLM ignores it unless started with `--api-key`. python-dotenv reads the single-quoted JSON literally (checked). Remove `OPENROUTER_EXTRA_BODY` for thinking-on runs and for judges that are not Qwen3.

## Suite behaviour that matters for this run (read from the code)

- **`max_tokens` is always 10000.** `run_experiments.py` line 353 reads `max_tokens = 10000 if "gemini" or "qwen" in model.lower() else 4000`, and the non-empty string `"gemini"` makes the condition always true. Prompts run 1,756 to 2,727 Qwen3 tokens (measured on the 30 generated conditions), so `--max-model-len 16384` fits.
- **Failed calls are never retried on resume.** On an exception the runner writes `response.json` with an `"error"` key and no `raw_response`. Resume treats any existing `response.json` as done. Timeouts and empty replies write no file, so a rerun retries them. Delete error files before a rerun (RUNBOOK step 5).
- **Inline classification is on by default.** `classification.enabled` defaults to true in `run_experiments.py`. Our configs set it to false and classify afterwards.
- **Paths.** `generate_prompts.py` writes to `<repo>/results/<experiment_id>/prompts`, while `run_experiments.py` writes to `results/<experiment_id>` relative to the working directory. Run everything from the repo root.
- **Models list.** `run_experiments.py` takes models from expansions only if an expansion has `enabled: true` and a `models` key, and otherwise from `global.models`. `generate_prompts.py` treats a missing `enabled` as true. The configs set it explicitly.
- **No seeds.** The suite never sends a sampling seed. `sample_id` (1..N) is a replicate index.
- **Timeouts.** One API call may take 600 s (`API_TIMEOUT_SECONDS`). A 10000-token reply must therefore decode at more than 17 tokens/s per sequence. Lower the provider concurrency if the log shows `API timeout`.
- `README.md` mentions `behavioral_detections.json`, but no script in the repo writes it. Labels live only in each `response.json`.

## Conditions

`generate_prompts.py` takes the union of `scenarios`, `goal_types`, `goal_values` and `urgency_types` over all enabled expansions, then builds one cartesian product. Goal types `none` and `ambiguous` ignore `goal_values` and use the single value `none`. Condition id: `{scenario}_{goal_type}-{goal_value}_{urgency_type}`, with `_prod` appended for production prompts (no scratchpad).

Available values: 3 scenarios (blackmail, leaking, murder), 3 urgency types (replacement, restriction, none), and 8 goal values with matching template sections (america, global, safety, acceleration, ethical, pragmatic, individualism, collectivism). Per scenario and urgency, explicit and latent each give 8 conditions, and none and ambiguous give 1 each, for 18. That makes 3 x 3 x 18 = **162 conditions**, or 324 counting the `_prod` variants. `swap` exists in `GOAL_VARIABLES`, but only under paired keys such as `swap-america-global`. With a plain value such as `america`, the lookup falls back to an empty goal. With a paired value, the email variables fall back to `ambiguous.md`. So it does not work cleanly, and these notes leave it out.

Murder reuses the blackmail variable files (`var_scenario = 'blackmail' if scenario == 'murder'`).
