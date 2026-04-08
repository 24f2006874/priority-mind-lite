# 🧠 PriorityMind-Lite

**LLM-Rewarded Customer Support Ticket Triage Environment**

*Meta PyTorch OpenEnv Hackathon 2026 | Team Axiom (IIT Madras)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-green.svg)](https://github.com/meta-openenv/openenv-core)

---

## 🎯 One-Line Pitch

An OpenEnv reinforcement learning environment where AI agents learn to triage customer support tickets using rewards evaluated by Llama — not hardcoded rules — enabling nuanced behaviors like empathy, strategic escalation, and contextual judgment.

## ✨ Key Innovation

Traditional RL requires you to define "good" mathematically. **How do you encode empathy as a number?** You don't — you let Llama judge it.

| Traditional RL | PriorityMind-Lite |
|----------------|-------------------|
| Hardcoded rewards (+10 resolution, -5 escalation) | LLM evaluates on empathy, efficiency, strategy |
| Optimizes numbers, not behavior | Learns human-like judgment |
| Fails at nuanced tasks | Handles contextual appropriateness |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run mock benchmark (no API key needed)
python inference.py --mock --verbose

# Run with live LLM evaluation (requires HF_TOKEN)
export HF_TOKEN=hf_your_token_here
python inference.py --verbose

# Run interactive demo for judges
python demo.py --live

# Launch Gradio web interface
python app.py
```

## 📋 What's Included

### Core Components

| File | Description |
|------|-------------|
| `environment.py` | PriorityMindEnv - OpenEnv-compatible environment |
| `grader.py` | HybridGrader (LLM + Fallback) + ProgrammaticGrader |
| `inference.py` | Benchmark runner with [START]/[STEP]/[END] format |
| `models.py` | Pydantic-typed Observation, Action, Reward models |
| `demo.py` | 90-second judge-facing demo script |
| `app.py` | Gradio web interface for HF Spaces |

### Supporting Files

| File | Description |
|------|-------------|
| `openenv.yaml` | OpenEnv metadata specification |
| `Dockerfile` | HF Spaces compatible container |
| `requirements.txt` | Pinned dependencies |
| `RESULTS.md` | Baseline benchmark results |
| `LICENSE` | MIT License |

### Validation & Testing

| File | Description |
|------|-------------|
| `scripts/validate_submission.py` | Pre-submission validation script |
| `tests/test_environment.py` | Comprehensive test suite |

## 🎮 Task Definitions

Each difficulty level includes **6 different ticket variations** to simulate real-world diversity. The environment randomly selects a variation on each reset, ensuring robust training and evaluation.

### Task 1: Easy — Simple Billing Inquiries (6 variations)
- **Sample Tickets**:
  - "My bill is higher than expected"
  - "I was charged twice for my subscription"
  - "Can you explain the charges on my latest invoice?"
  - "My payment failed but I was still charged"
  - "I need a refund for the overcharge on my account"
  - "Why did my monthly fee increase without notice?"
- **Sentiment Range**: -0.1 to -0.5 (mildly negative)
- **True Category**: billing
- **Max Steps**: 3
- **Success**: Correctly categorize + assign appropriate priority

### Task 2: Medium — Frustrated Technical Issues (6 variations)
- **Sample Tickets**:
  - "App keeps crashing! I'm so frustrated!"
  - "The app freezes every time I try to upload a photo"
  - "Login page shows error 500 constantly"
  - "My notifications stopped working after the update"
  - "The search function returns no results even for items I know exist"
  - "App drains my battery in just 2 hours"
- **Sentiment Range**: -0.65 to -0.8 (frustrated)
- **True Category**: technical
- **Max Steps**: 5
- **Success**: Recognize frustration, prioritize high/urgent, empathetic response

### Task 3: Hard — Complex Multi-Issue Complaints (6 variations)
- **Sample Tickets**:
  - "I've waited 3 days for a refund AND your app deleted my data. This is unacceptable!"
  - "Your service has been down for 2 days and I'm losing business. I want compensation!"
  - "I've been transferred 5 times and no one has solved my problem. This is terrible service!"
  - "My account was hacked and your support team is not responding fast enough!"
  - "You charged me for a year subscription but I only wanted monthly. Refund the difference NOW!"
  - "My personal data was exposed in your data breach and I haven't heard from you in a week!"
- **Sentiment Range**: -0.88 to -0.97 (very angry)
- **True Category**: complaint
- **Max Steps**: 8
- **Success**: De-escalate, coordinate multi-step resolution, maintain empathy

## 🏆 Benchmark Results

| Task   | Avg Score | Empathy | Efficiency | Strategy | Resolved |
|--------|-----------|---------|------------|----------|----------|
| easy   | 0.30      | 0.30/1.0 | 0.50/1.0   | 0.77/1.0 | ✅ Yes   |
| medium | 0.24      | 0.44/1.0 | 0.57/1.0   | 0.71/1.0 | ✅ Yes   |
| hard   | 0.20      | 0.41/1.0 | 0.56/1.0   | 0.70/1.0 | ✅ Yes   |
| **MEAN** | **0.25** | **0.38/1.0** | **0.54/1.0** | **0.73/1.0** | - |

Verified with `python inference.py --mock --verbose` using the seeded offline benchmark. Live scores depend on the configured HF model and fallback rate, so only the deterministic mock baseline is pinned in the repo.

See [RESULTS.md](RESULTS.md) for detailed benchmark methodology and the verified mock baseline.

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL FLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Environment Reset                                       │
│     • Generate ticket: text + sentiment + true_category     │
│     • Return typed Observation (Pydantic model)             │
│                                                             │
│  2. Agent Takes Action                                      │
│     • Uses OpenAI-compatible client (HF Router)            │
│     • Action types: categorize/prioritize/respond/escalate/resolve │
│                                                             │
│  3. Hybrid Grader Evaluates                                 │
│     ├─ Try LLM Evaluation (60% of reward)                  │
│     │  • Prompt Llama with state + action + criteria       │
│     │  • Llama returns: score (0-10) + reasoning + sub-scores│
│     │  • Normalize to [0.0, 1.0]                           │
│     │                                                       │
│     ├─ Cache Result (exact-match caching)                   │
│     │  • Reduces API calls for repeated queries            │
│     │                                                       │
│     └─ Fallback to Programmatic (40% of reward)             │
│        • Deterministic rules if LLM fails                  │
│                                                             │
│  4. Return Reward + New State + Done Flag                   │
│     • Reward = 0.4*programmatic + 0.6*llm_normalized_score │
│     • Partial signals: empathy, efficiency, strategy        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🌐 Live Demo

Try the interactive demo on Hugging Face Spaces:

👉 **[https://huggingface.co/spaces/TeamAxiom/priority-mind-lite](https://huggingface.co/spaces/TeamAxiom/priority-mind-lite)**

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API key | (required for live mode) |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct:novita` |
| `HF_TIMEOUT_SECONDS` | API timeout | `12` |

### .env File

```env
# Hugging Face API Configuration
HF_TOKEN=hf_your_token_here

# API Configuration - Use HF Router for Inference
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct:novita
HF_TIMEOUT_SECONDS=12
```

## 🧪 Testing & Validation

```bash
# Run all tests
python -m pytest tests/ -v

# Run pre-submission validation
python scripts/validate_submission.py

# Validate OpenEnv spec
openenv validate . --verbose

# Run mock inference (validates output format)
python inference.py --mock
```

## 📁 Project Structure

```
priority-mind-lite/
├── openenv.yaml              # OpenEnv metadata
├── environment.py            # PriorityMindEnv class
├── grader.py                 # HybridGrader + ProgrammaticGrader
├── inference.py              # Benchmark runner
├── models.py                 # Pydantic models
├── demo.py                   # Judge-facing demo
├── app.py                    # Gradio web interface
├── Dockerfile                # Container configuration
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
├── README.md                 # This file
├── RESULTS.md                # Benchmark results
├── .huggingface/             # HF Spaces config
│   └── README.md
├── scripts/
│   └── validate_submission.py
├── server/
│   └── app.py                # OpenEnv HTTP server
└── tests/
    └── test_environment.py
```

## 🎬 90-Second Demo Flow

1. **Problem Statement** (15s): "How do you mathematically define empathy?"
2. **Solution** (15s): "Let Llama judge what good looks like"
3. **Live Demo** (30s): Run hard task with visible LLM reasoning
4. **Results** (15s): Show metrics table with empathy improvement
5. **Insight** (15s): "We taught the agent to satisfy a judge that values empathy"

Run with: `python demo.py --live`

## 🏅 Why This Wins

1. **Perfect Requirement Alignment**: Customer support explicitly allowed, 3 tasks, <20min runtime, exact output format
2. **Research-Grade Innovation**: First OpenEnv environment with LLM-as-reward-function
3. **Meta Stack Alignment**: OpenEnv + Llama + PyTorch + HF Spaces
4. **Demo-Ready Design**: Visible LLM reasoning, clear metrics, memorable takeaway
5. **Production-Thinking**: Fallback grader, smart caching, pinned dependencies

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Team

**Team Axiom** - IIT Madras
- Rauank Ratan
- Akash Deep
- Sangam Jha

## 🔗 Links

- [GitHub Repository](https://github.com/TeamAxiom/priority-mind-lite)
- [HF Spaces Demo](https://huggingface.co/spaces/TeamAxiom/priority-mind-lite)
- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [Hackathon Portal](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)

---

*Built on OpenEnv. Judged by Llama.*
