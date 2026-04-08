---
title: PriorityMind-Lite
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
license: mit
tags:
  - openenv
  - customer-support
  - llm-reward
  - reinforcement-learning
  - meta-pytorch-hackathon
---

# 🧠 PriorityMind-Lite

**LLM-Rewarded Customer Support Ticket Triage Environment**

*Meta PyTorch OpenEnv Hackathon 2026 | Team Axiom (IIT Madras)*

## What is this?

PriorityMind-Lite is an OpenEnv reinforcement learning environment where AI agents learn to triage customer support tickets using rewards evaluated by Llama — not hardcoded rules. This enables nuanced behaviors like empathy, strategic escalation, and contextual judgment.

## Key Innovation

Instead of defining "good customer service" mathematically (+10 for resolution, -5 for escalation), we let a language model (Llama) judge each agent action on dimensions like empathy, efficiency, and strategy.

## How to Use

1. **Interactive Demo**: Go to the "Interactive Demo" tab
2. **Select Task Difficulty**: Choose easy, medium, or hard
3. **Toggle LLM Evaluation**: Enable for live Llama judging (requires HF_TOKEN secret)
4. **Run Demo**: Watch the agent triage the ticket step-by-step
5. **View Results**: See scores, reasoning, and partial signals

## Technology

- **Framework**: OpenEnv (Meta)
- **Model**: Llama 3.1 8B Instruct (via HF Router)
- **Interface**: Gradio
- **Language**: Python 3.10

## Links

- [GitHub Repository](https://github.com/24f2006874/priority-mind-lite)
- [Hackathon Submission](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)

---

*Built on OpenEnv. Judged by Llama.*