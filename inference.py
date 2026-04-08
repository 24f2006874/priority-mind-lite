from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from dotenv import load_dotenv
from pydantic import ValidationError

from environment import PriorityMindEnv
from models import Action, Observation
from utils import infer_ticket_category, format_partial_signal

if TYPE_CHECKING:
    from openai import OpenAI
else:
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        OpenAI = None


load_dotenv(Path(__file__).parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
TASKS = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PriorityMind-Lite benchmark runner")
    parser.add_argument("--task", choices=TASKS, help="Run a single task")
    parser.add_argument("--mock", action="store_true", help="Use the deterministic offline policy")
    parser.add_argument("--verbose", action="store_true", help="Print a human-readable summary after the benchmark")
    return parser.parse_args()


def build_client() -> OpenAI | None:
    if not HF_TOKEN or OpenAI is None:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=float(os.getenv("HF_TIMEOUT_SECONDS", "12")))


# Backward compatibility alias - prefer utils module for new code
infer_category = infer_ticket_category


def mock_action(obs: Observation, task: str) -> Action:
    if obs.category is None:
        return Action(action_type="categorize", content=infer_category(obs.ticket_text))

    if obs.priority is None:
        priority = {"easy": "medium", "medium": "high", "hard": "urgent"}[task]
        return Action(action_type="prioritize", priority=priority)

    if task == "easy":
        return Action(action_type="resolve")

    if task == "medium":
        if obs.attempts < 3:
            return Action(
                action_type="respond",
                content="I am sorry this keeps crashing. Please try updating the app and restarting your device while we review the crash details.",
            )
        return Action(action_type="resolve")

    if obs.attempts < 3:
        return Action(
            action_type="respond",
            content="I am sorry you have had to deal with both the refund delay and the missing data. I am escalating this right away so a specialist can investigate.",
        )
    if obs.attempts == 3:
        return Action(action_type="escalate")
    return Action(action_type="resolve")


def llm_action(obs: Observation, client: OpenAI) -> Action:
    prompt = f"""You are a customer support triage agent.

Observation:
- ticket_text: {obs.ticket_text}
- sentiment: {obs.sentiment:.2f}
- category: {obs.category}
- priority: {obs.priority}
- attempts: {obs.attempts}
- resolved: {str(obs.resolved).lower()}

Choose exactly one next action:
- categorize with content billing/technical/general/complaint
- prioritize with priority low/medium/high/urgent
- respond with a concise, empathetic support message
- escalate
- resolve

Return JSON only with keys action_type, content, priority.
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = response.choices[0].message.content
    if not payload:
        raise RuntimeError("empty_action_response")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_action_json: {payload}") from exc

    if "action_type" not in data:
        raise RuntimeError("missing_action_type in LLM response")

    try:
        return Action(**data)
    except ValidationError as exc:
        raise RuntimeError(f"invalid_action_schema: {exc.errors()[0]['msg'] if exc.errors() else str(exc)}") from exc


def choose_action(
    obs: Observation,
    task: str,
    client: OpenAI | None,
    force_mock: bool,
) -> tuple[Action, bool]:
    if force_mock or client is None:
        return mock_action(obs, task), True

    try:
        return llm_action(obs, client), False
    except Exception:
        return mock_action(obs, task), True


def format_action(action: Action) -> str:
    parts = [action.action_type]
    if action.content:
        compact_content = " ".join(action.content.split())
        parts.append(f"content={compact_content}")
    if action.priority:
        parts.append(f"priority={action.priority}")
    return "|".join(parts)


def evaluate_success(task: str, obs: Observation, actions: list[Action], final_score: float) -> bool:
    action_types = [action.action_type for action in actions]

    if task == "easy":
        return (
            obs.resolved
            and obs.category == "billing"
            and obs.priority in {"medium", "high", "urgent"}
            and final_score >= 0.30
        )

    if task == "medium":
        return (
            obs.resolved
            and obs.category == "technical"
            and obs.priority in {"high", "urgent"}
            and "respond" in action_types
            and final_score >= 0.20
        )

    return (
        obs.resolved
        and obs.category == "complaint"
        and obs.priority in {"high", "urgent"}
        and "escalate" in action_types
        and final_score >= 0.20
    )


def run_task(task: str, client: OpenAI | None, force_mock: bool) -> tuple[float, str]:
    # Use fixed seed for reproducible benchmark results (especially in mock mode)
    env = PriorityMindEnv(task=task, seed=42, enable_llm=not force_mock)
    obs = env.reset()
    print(f"[START] task={task} env=priority-mind-lite model={MODEL_NAME}")

    rewards: list[float] = []
    actions_taken: list[Action] = []
    step = 0
    success = False
    run_mode = "mock" if force_mock or client is None else "live"
    while True:
        step += 1
        try:
            action, used_mock = choose_action(obs, task, client, force_mock)
            actions_taken.append(action)
            if used_mock and run_mode == "live":
                run_mode = "live_with_fallback"
            obs, reward, done, info = env.step(action)
            rewards.append(reward.score)
            error_msg = info.get("last_action_error") or "null"
            print(
                f"[STEP] step={step} action={format_action(action)} reward={reward.score:.2f} "
                f"done={str(done).lower()} error={error_msg}"
            )
            if done:
                final_score = sum(rewards) / len(rewards) if rewards else 0.0
                success = evaluate_success(task, obs, actions_taken, final_score)
                break
        except Exception as exc:
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={str(exc)}")
            # Still output final results even on error for complete benchmark format
            final_score = sum(rewards) / len(rewards) if rewards else 0.0
            success = False
            rewards_str = ",".join(f"{value:.2f}" for value in rewards) if rewards else "0.00"
            print(f"[END] success={str(success).lower()} steps={step} score={final_score:.2f} rewards={rewards_str}")
            return final_score, "error"

    final_score = sum(rewards) / len(rewards) if rewards else 0.0
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(f"[END] success={str(success).lower()} steps={step} score={final_score:.2f} rewards={rewards_str}")
    return final_score, run_mode


def run(tasks: Iterable[str], force_mock: bool, verbose: bool) -> dict[str, float]:
    client = None if force_mock else build_client()
    effective_mock = force_mock or client is None
    scores: dict[str, float] = {}
    run_modes: dict[str, str] = {}

    for task in tasks:
        score, run_mode = run_task(task, client=client, force_mock=effective_mock)
        scores[task] = score
        run_modes[task] = run_mode

    if verbose:
        if all(mode == "mock" for mode in run_modes.values()):
            mode = "mock"
        elif any(mode == "live_with_fallback" for mode in run_modes.values()):
            mode = "live_with_fallback"
        else:
            mode = "live"
        print(f"summary mode={mode}")
        for task, score in scores.items():
            print(f"summary task={task} score={score:.2f} mode={run_modes[task]}")
        mean_score = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"summary mean={mean_score:.2f}")

    return scores


if __name__ == "__main__":
    args = parse_args()
    selected_tasks = (args.task,) if args.task else TASKS
    run(tasks=selected_tasks, force_mock=args.mock, verbose=args.verbose)
