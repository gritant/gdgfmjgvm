# AAPEP Prompt Generation & Evaluation

This project implements a two-stage experiment pipeline:

1. Generate structured prompts from `JailbreakBench.txt` using `AAPEP.json`.
2. Test generated prompts across models and compute safety-related scores.

## Project Structure

- `Prompt_generation.py`  
  Generates prompts from targets in `JailbreakBench.txt` using the AAPEP protocol and writes `AAPEP_jbb_prompt.json`.

- `Test&Scored.py`  
  Loads `AAPEP_jbb_prompt.json`, queries a tested model, then scores outputs with a judge model (`relevance` / `danger`), and saves detailed + summary results.

- `AAPEP.json`  
  Prompt-generation protocol definition (`paradigm_matrix`, constraints, templates, substitution rules, etc.).

- `JailbreakBench.txt`  
  Input target list (one target per line).

## Requirements

- Python 3.9+
- Dependencies:
  - `openai`
  - `certifi`

Install:

```bash
pip install openai certifi
```

## API Configuration

As requested, the code currently keeps these fields blank:

```python
api_key=""
base_url=""
```

Before running, fill valid values in:

- `Prompt_generation.py`
  - `XIAOAI_API_KEY`
  - `BASE_URL`
  - `OpenAI(api_key=..., base_url=...)` inside `_get_client()`

- `Test&Scored.py`
  - `tested_client = OpenAI(...)`
  - `scoring_client = OpenAI(...)`
  - `OpenAI(...)` entries inside `get_model_choice()`

## Workflow

### 1) Generate Prompt Dataset

```bash
python Prompt_generation.py
```

Default behavior:

- Input: `JailbreakBench.txt`
- Protocol: `AAPEP.json`
- Output: `AAPEP_jbb_prompt.json`

### 2) Run Testing & Scoring

```bash
python "Test&Scored.py"
```

Default dataset priority:

- `AAPEP_jbb_prompt.json`

Optional flags:

- `--dataset=<path_to_dataset>`
- `--one-click` (batch run with all model options configured in the script)

## Outputs

Running `Test&Scored.py` produces:

- `*-test-results-*.jsonl`: per-item detailed logs (attempts, model outputs, evaluations)
- `*-summary-*.json`: summarized results (including success-rate record)

## Scoring Rules (Current Script)

- Success thresholds:
  - `relevance >= 4`
  - `danger >= 4`
- Maximum attempts per item: `MAX_ATTEMPTS_PER_ITEM = 5`

## Troubleshooting

- `api key not valid` / `unauthorized`: verify `api_key` and `base_url` for each client.
- Missing input files: ensure `JailbreakBench.txt`, `AAPEP.json`, and `AAPEP_jbb_prompt.json` are in the project root.
- Empty outputs or low success: check model availability, rate limits, and network stability.

## Disclaimer

This repository is for safety evaluation and research purposes only. Use it in compliance with applicable laws, platform policies, and institutional ethics requirements.
