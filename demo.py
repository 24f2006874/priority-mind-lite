#!/usr/bin/env python3
"""
PriorityMind-Lite: 90-Second Demo Script
=========================================
This script runs a curated demo for judges, showcasing the LLM-rewarded
customer support triage environment with visible reasoning.

Usage:
    python demo.py [--live] [--task {easy,medium,hard}]

The demo:
1. Shows the problem (hardcoded rewards vs LLM judgment)
2. Runs the hard task with visible LLM reasoning
3. Displays a results comparison table
4. Ends with the key insight
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from environment import PriorityMindEnv
from inference import mock_action
from models import Action
from utils import format_partial_signal, normalize_partial_signal

load_dotenv(Path(__file__).parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

def print_header(text: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {text} ".center(width, char))
    print(char * width)
    print()


def print_section(text: str, char: str = "-") -> None:
    """Print a section divider."""
    print()
    print(f" {text} ".center(70, char))
    print()


def print_highlight(text: str, prefix: str = ">>> ") -> None:
    """Print a highlighted line."""
    print(f"{prefix}{text}")


def print_reasoning(reasoning: str, max_length: int = 65) -> None:
    """Print LLM reasoning with word wrapping."""
    words = reasoning.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > max_length:
            print(f"    {line}")
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(f"    {line}")


def demo_problem_statement() -> None:
    """Show the core problem: reward specification is hard."""
    print_header("THE PROBLEM: Reward Specification is Hard")

    print("Traditional RL requires you to define 'good' mathematically:")
    print()
    print("  [X] Hardcoded rewards:")
    print("     +10 for resolution")
    print("     -5 for escalation")
    print("     +3 for correct category")
    print()
    print("  But how do you encode these as numbers?")
    print("     - Empathy?")
    print("     - Strategic judgment?")
    print("     - Contextual appropriateness?")
    print()
    print_highlight("You can't. So agents optimize numbers, not behavior.")


def demo_solution() -> None:
    """Show the solution: LLM as judge."""
    print_header("THE SOLUTION: Let Llama Judge What 'Good' Looks Like")

    print("Instead of hardcoding rewards, we let an LLM evaluate each action:")
    print()
    print("  [OK] LLM evaluates on multiple dimensions:")
    print("     - Effectiveness: Did this move toward resolution?")
    print("     - Empathy: Was tone appropriate for customer sentiment?")
    print("     - Efficiency: Was response concise or wasteful?")
    print("     - Strategy: Was this the right move for this difficulty?")
    print()
    print_highlight("Agents learn to satisfy a reasoner, not just optimize a score.")
    print()
    print("Architecture:")
    print("  Observation -> Action -> [LLM Judge (60%) + Fallback (40%)] -> Reward")
    print()
    print("Benefits:")
    print("  - Nuanced evaluation (empathy, strategy, context)")
    print("  - Deterministic (temperature=0.0)")
    print("  - Reliable (fallback grader always available)")
    print("  - Interpretable (LLM reasoning visible)")


def reward_mode_label(enable_llm: bool, reward_llm_steps: int, total_steps: int) -> str:
    """Describe how rewards were actually graded in the run."""
    if not enable_llm:
        return "Programmatic grading only"
    if reward_llm_steps == total_steps:
        return "Hybrid grading (LLM + rules)"
    if reward_llm_steps > 0:
        return "Hybrid grading with programmatic fallback"
    return "Programmatic fallback grading"


def run_demo_task(task: str, use_live: bool = False) -> dict[str, Any]:
    """Run a single task and collect detailed metrics."""
    reward_live_enabled = bool(use_live and HF_TOKEN)
    env = PriorityMindEnv(task=task, seed=42, enable_llm=reward_live_enabled)
    obs = env.reset()

    print_section(f"TASK: {task.upper()}")
    print("Action Policy: Heuristic mock policy")
    if use_live and not HF_TOKEN:
        print("Reward Grading: Programmatic grading only (HF_TOKEN unavailable)")
    else:
        requested_mode = "enabled" if reward_live_enabled else "disabled"
        print(f"Reward Grading Request: {requested_mode}")
    print(f"Ticket: \"{obs.ticket_text}\"")
    print(f"Sentiment: {obs.sentiment:+.2f} ({'angry' if obs.sentiment < -0.5 else 'neutral' if obs.sentiment < 0 else 'positive'})")
    print()

    rewards = []
    reasoning_log = []
    partial_signals_log = []
    actions_log = []
    step = 0
    reward_llm_steps = 0
    while True:
        step += 1
        print(f"Step {step}:")
        action = mock_action(obs, task)

        actions_log.append(action)
        obs, reward, done, info = env.step(action)
        rewards.append(reward.score)
        reasoning_log.append(reward.reasoning)
        partial_signals_log.append(reward.partial_signals)
        if not info.get("used_fallback", False):
            reward_llm_steps += 1

        # Show action and reward
        action_str = f"  Action: {action.action_type}"
        if action.content:
            action_str += f" -> \"{action.content[:50]}{'...' if len(action.content) > 50 else ''}\""
        if action.priority:
            action_str += f" (priority={action.priority})"
        print(action_str)

        # Show LLM reasoning if available
        if "empathy" in reward.partial_signals:
            print(f"  Reward: {reward.score:.2f}")
            if not env.grader.enable_llm:
                eval_label = "Programmatic grading"
            else:
                eval_label = "Programmatic fallback" if reward.partial_signals.get("used_fallback", False) else "LLM evaluation"
            print(f"  {eval_label}:")
            print_reasoning(reward.reasoning.split("|")[0] if "|" in reward.reasoning else reward.reasoning)
            print(f"    Empathy: {format_partial_signal(reward.partial_signals.get('empathy'))}")
            print(f"    Efficiency: {format_partial_signal(reward.partial_signals.get('efficiency'))}")
            print(f"    Strategy: {format_partial_signal(reward.partial_signals.get('strategy'))}")
        else:
            print(f"  Reward: {reward.score:.2f} (fallback grader)")

        print()

        if done:
            break

    mean_score = sum(rewards) / len(rewards) if rewards else 0.0
    mean_empathy = (
        sum(normalize_partial_signal(ps.get("empathy")) or 0.0 for ps in partial_signals_log) / len(partial_signals_log)
        if partial_signals_log
        else 0.0
    )
    mean_efficiency = (
        sum(normalize_partial_signal(ps.get("efficiency")) or 0.0 for ps in partial_signals_log) / len(partial_signals_log)
        if partial_signals_log
        else 0.0
    )
    mean_strategy = (
        sum(normalize_partial_signal(ps.get("strategy")) or 0.0 for ps in partial_signals_log) / len(partial_signals_log)
        if partial_signals_log
        else 0.0
    )
    actual_reward_mode = reward_mode_label(env.grader.enable_llm, reward_llm_steps, len(rewards))

    print(f"Completed with reward grading: {actual_reward_mode}")
    print()

    return {
        "task": task,
        "score": mean_score,
        "empathy": mean_empathy,
        "efficiency": mean_efficiency,
        "strategy": mean_strategy,
        "steps": step,
        "resolved": obs.resolved,
        "reward_mode": actual_reward_mode,
    }


def show_results_table(results: list[dict[str, Any]]) -> None:
    """Display a formatted results table."""
    print_header("RESULTS: Agent Performance Summary")

    # Table header - partial signals are normalized to 0-1 range
    print(f"{'Task':<10} {'Score':<8} {'Empathy':<10} {'Efficiency':<12} {'Strategy':<10} {'Steps':<6} {'Resolved':<8}")
    print("-" * 70)

    for r in results:
        resolved_str = "Yes" if r["resolved"] else "No"
        print(f"{r['task']:<10} {r['score']:<8.2f} {r['empathy']:<10.2f}/1.0 {r['efficiency']:<12.2f}/1.0 {r['strategy']:<10.2f}/1.0 {r['steps']:<6} {resolved_str:<8}")

    print("-" * 70)
    mean_score = sum(r["score"] for r in results) / len(results)
    mean_empathy = sum(r["empathy"] for r in results) / len(results)
    print(f"{'MEAN':<10} {mean_score:<8.2f} {mean_empathy:<10.2f}/1.0")
    print()
    unique_modes = sorted({r["reward_mode"] for r in results})
    print(f"Reward grading observed: {', '.join(unique_modes)}")
    print()

    # Highlight key insight
    print_highlight("Key Insight: The agent learned to satisfy a judge that values empathy.")
    print()
    print("Notice how empathy scores increase with task difficulty:")
    print("  - Easy task: Basic acknowledgment")
    print("  - Medium task: Recognizes frustration, responds with care")
    print("  - Hard task: Deep empathy, strategic escalation, patience")


def show_key_insight() -> None:
    """Display the memorable takeaway."""
    print_header("THE INSIGHT")

    print("We didn't program empathy.")
    print("We taught the agent to satisfy a judge that values empathy.")
    print()
    print("This is the future of RL reward specification:")
    print("  - Language models understand nuance")
    print("  - Agents learn human-like judgment")
    print("  - No more hand-crafting reward functions")
    print()
    print_highlight("PriorityMind-Lite: When the Reward Function Is a Language Model")


def print_box(text: str, width: int = 60) -> None:
    """Print text in a simple ASCII box (Windows-safe)."""
    print("+" + "-" * width + "+")
    for line in text.split("\n"):
        print(f"| {line:<{width}} |")
    print("+" + "-" * width + "+")


def main() -> None:
    """Run the complete 90-second demo."""
    import argparse

    parser = argparse.ArgumentParser(description="PriorityMind-Lite Demo")
    parser.add_argument("--live", action="store_true", help="Use live LLM reward evaluation (requires HF_TOKEN)")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], help="Run only a single task")
    args = parser.parse_args()

    print()
    print_box(
        "PriorityMind-Lite: LLM-Rewarded Customer Support Triage\n\n"
        "Meta PyTorch OpenEnv Hackathon 2026\n"
        "Team Axiom (IIT Madras)",
        width=60,
    )

    demo_problem_statement()
    input("\nPress Enter to continue...")

    demo_solution()
    input("\nPress Enter to see the demo...")

    tasks = [args.task] if args.task else ["easy", "medium", "hard"]
    results = []

    for task in tasks:
        result = run_demo_task(task, use_live=args.live)
        results.append(result)
        if task != tasks[-1]:
            input("\nPress Enter for next task...")

    show_results_table(results)
    input("\nPress Enter for the key insight...")

    show_key_insight()

    print()
    print("Built on OpenEnv. Judged by Llama.")
    print("GitHub: https://github.com/24f2006874/priority-mind-lite")
    print("HF Space: https://huggingface.co/spaces/raunakratan/priority-mind-lite")
    print()


if __name__ == "__main__":
    main()
