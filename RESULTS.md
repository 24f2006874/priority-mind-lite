# PriorityMind-Lite: Baseline Benchmark Results

This document reports the verified seeded baseline for PriorityMind-Lite using the deterministic mock policy. These numbers are reproducible locally and are the benchmark figures pinned in the repository.

## Benchmark Configuration

- **Mode**: Mock (`--mock`)
- **Policy**: Deterministic heuristic policy from `inference.py`
- **Environment**: `PriorityMindEnv(task=..., seed=42, enable_llm=False)`
- **Command**: `python inference.py --mock --verbose`
- **Runtime**: Single-digit seconds for all three tasks on a typical laptop

## Verified Mock Baseline

| Task   | Avg Score | Empathy | Efficiency | Strategy | Steps | Resolved |
|--------|-----------|---------|------------|----------|-------|----------|
| easy   | 0.30      | 0.30/1.0 | 0.50/1.0 | 0.77/1.0 | 3     | Yes      |
| medium | 0.24      | 0.44/1.0 | 0.57/1.0 | 0.71/1.0 | 4     | Yes      |
| hard   | 0.20      | 0.41/1.0 | 0.56/1.0 | 0.70/1.0 | 5     | Yes      |
| **MEAN** | **0.25** | **0.38/1.0** | **0.54/1.0** | **0.73/1.0** | - | - |

## Reproduce Locally

```bash
pip install -r requirements.txt
python inference.py --mock --verbose
```

You should see:

- `summary task=easy score=0.30 mode=mock`
- `summary task=medium score=0.24 mode=mock`
- `summary task=hard score=0.20 mode=mock`
- `summary mean=0.25`

## Live Mode Notes

Live mode is intentionally not pinned to a single score table in the repository.

Why:

- The configured model can change.
- HF Router availability can vary.
- The hybrid grader may fall back per step.
- Even when the model is fixed, infrastructure behavior can affect the observed mix of LLM and fallback grading.

To inspect a live run locally:

```bash
python inference.py --verbose
```

If the live run falls back during action generation, the benchmark reports `mode=live_with_fallback`. In the app and CLI demo, action selection and reward grading are now reported separately so fallback is explicit.

## Observations

1. The offline benchmark is deterministic and suitable for CI validation.
2. Easy tickets score highest because the task can complete after basic triage.
3. Medium and hard tasks still resolve successfully, but their stricter reward criteria lower the average score.
4. Partial signals are normalized to the `0.0-1.0` range for both UI and demo display.

---

*Results generated from the current seeded mock baseline on April 8, 2026.*
