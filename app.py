"""
PriorityMind-Lite: Gradio Web Interface for Hugging Face Spaces
================================================================
This provides an interactive demo of the PriorityMind-Lite environment
where users can interact with the AI customer support triage agent.

Deployed at: https://huggingface.co/spaces/TeamAxiom/priority-mind-lite
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

import gradio as gr
from dotenv import load_dotenv

from environment import PriorityMindEnv
from inference import mock_action
from models import Action, Observation
from utils import format_partial_signal

load_dotenv(Path(__file__).parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Try to import openai for live mode
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


APP_THEME = gr.themes.Soft()


def get_client() -> OpenAI | None:
    """Get OpenAI client for HF Router if available."""
    if not HF_TOKEN or not OPENAI_AVAILABLE:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=12)


def configure_runtime() -> None:
    """Use a Windows-friendly event loop policy for Gradio/Uvicorn."""
    if os.name != "nt":
        return
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except (AttributeError, RuntimeError):
        pass


def _llm_prompt(obs: Observation) -> str:
    """Build the LLM prompt for action selection."""
    return f"""You are a customer support triage agent.

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


def choose_action(obs: Observation, task: str, client: OpenAI | None) -> tuple[Action, str]:
    """Choose the next action using LLM if available, else use heuristic.

    Returns:
        A tuple of (action, action_source) where action_source is one of:
        "LLM", "heuristic", or "heuristic fallback".
    """
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": _llm_prompt(obs)}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            payload = response.choices[0].message.content
            if payload:
                data = json.loads(payload)
                action = Action(**data)
                return action, "LLM"
        except Exception:
            return mock_action(obs, task), "heuristic fallback"

    return mock_action(obs, task), "heuristic"


def format_action(action: Action) -> str:
    """Format action for display."""
    parts = [action.action_type.upper()]
    if action.content:
        parts.append(f'"{action.content[:60]}{"..." if len(action.content) > 60 else ""}"')
    if action.priority:
        parts.append(f"priority={action.priority}")
    return " -> ".join(parts)


def _action_mode_label(mode: str) -> str:
    labels = {
        "llm_only": "LLM policy",
        "mixed": "LLM policy with heuristic fallback",
        "heuristic_only": "Heuristic policy",
        "heuristic_fallback": "Heuristic policy after LLM fallback",
    }
    return labels.get(mode, mode)


def _reward_mode_label(mode: str) -> str:
    labels = {
        "hybrid_llm": "Hybrid grading (LLM + rules)",
        "hybrid_mixed": "Hybrid grading with programmatic fallback",
        "programmatic_fallback": "Programmatic fallback grading",
        "programmatic_only": "Programmatic grading only",
    }
    return labels.get(mode, mode)


def run_episode(task: str, use_llm: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a complete episode and return step-by-step results plus mode summary."""
    client = get_client() if use_llm else None
    # Use a fixed seed for reproducible demo results
    env = PriorityMindEnv(task=task, seed=42, enable_llm=client is not None)
    obs = env.reset()

    steps = []
    step_num = 0
    llm_action_steps = 0
    reward_llm_steps = 0
    reward_fallback_steps = 0

    while True:
        step_num += 1
        action, action_source = choose_action(obs, task, client)
        if action_source == "LLM":
            llm_action_steps += 1

        obs, reward, done, info = env.step(action)
        reward_used_fallback = bool(info.get("used_fallback", False))
        if not env.grader.enable_llm:
            reward_source = "Programmatic grading"
        else:
            reward_source = "Programmatic fallback" if reward_used_fallback else "LLM judge"
        if reward_used_fallback:
            reward_fallback_steps += 1
        else:
            reward_llm_steps += 1

        step_info = {
            "step": step_num,
            "observation": {
                "ticket_text": obs.ticket_text,
                "sentiment": f"{obs.sentiment:+.2f}",
                "category": obs.category or "Not set",
                "priority": obs.priority or "Not set",
                "attempts": obs.attempts,
                "resolved": obs.resolved,
            },
            "action": format_action(action),
            "action_source": action_source,
            "action_error": info.get("last_action_error"),
            "reward": {
                "score": f"{reward.score:.2f}",
                "reasoning": reward.reasoning[:200] + "..." if len(reward.reasoning) > 200 else reward.reasoning,
            },
            "reward_source": reward_source,
            "partial_signals": {
                "empathy": format_partial_signal(reward.partial_signals.get("empathy")),
                "efficiency": format_partial_signal(reward.partial_signals.get("efficiency")),
                "strategy": format_partial_signal(reward.partial_signals.get("strategy")),
            },
            "done": done,
        }
        steps.append(step_info)

        if done:
            break

    if not use_llm or client is None:
        action_mode = "heuristic_only"
    elif llm_action_steps == len(steps):
        action_mode = "llm_only"
    elif llm_action_steps > 0:
        action_mode = "mixed"
    else:
        action_mode = "heuristic_fallback"

    if not env.grader.enable_llm:
        reward_mode = "programmatic_only"
    elif reward_llm_steps == len(steps):
        reward_mode = "hybrid_llm"
    elif reward_llm_steps > 0:
        reward_mode = "hybrid_mixed"
    else:
        reward_mode = "programmatic_fallback"

    summary = {
        "action_mode": action_mode,
        "reward_mode": reward_mode,
        "client_available": client is not None,
        "llm_requested": use_llm,
        "total_steps": len(steps),
        "llm_action_steps": llm_action_steps,
        "heuristic_action_steps": len(steps) - llm_action_steps,
        "reward_llm_steps": reward_llm_steps,
        "reward_fallback_steps": reward_fallback_steps,
    }
    return steps, summary


def run_demo(task: str, use_llm: bool) -> str:
    """Run demo and return formatted results."""
    try:
        steps, summary = run_episode(task, use_llm)

        output = []
        output.append(f"**Task: {task.upper()}**")
        output.append(f"**Action Policy:** {_action_mode_label(summary['action_mode'])}")
        output.append(f"**Reward Grading:** {_reward_mode_label(summary['reward_mode'])}")
        if use_llm and not summary["client_available"]:
            output.append("_Note: HF_TOKEN or the OpenAI client is unavailable, so the run used heuristic actions and programmatic grading only._")
        elif summary["action_mode"] == "mixed":
            output.append(
                f"_Action generation used the LLM in {summary['llm_action_steps']}/{summary['total_steps']} steps and heuristic fallback for the rest._"
            )
        elif summary["action_mode"] == "heuristic_fallback":
            output.append("_Every action fell back to the heuristic policy after an LLM generation failure._")

        if summary["reward_mode"] == "hybrid_mixed":
            output.append(
                f"_Reward grading used the LLM judge in {summary['reward_llm_steps']}/{summary['total_steps']} steps and programmatic fallback in the rest._"
            )
        elif summary["reward_mode"] == "programmatic_fallback":
            output.append("_Reward grading attempted live evaluation, but every step fell back to the programmatic grader._")
        output.append("")

        for step_info in steps:
            output.append(f"### Step {step_info['step']}")
            output.append(f"**Action:** {step_info['action']} _({step_info['action_source']})_")
            output.append(f"**Reward Source:** {step_info['reward_source']}")
            output.append(f"**Reward:** {step_info['reward']['score']}")
            output.append(f"**Reasoning:** {step_info['reward']['reasoning']}")
            output.append(f"- Empathy: {step_info['partial_signals']['empathy']}")
            output.append(f"- Efficiency: {step_info['partial_signals']['efficiency']}")
            output.append(f"- Strategy: {step_info['partial_signals']['strategy']}")
            if step_info["action_error"]:
                output.append(f"- Action Error: {step_info['action_error']}")
            if step_info['done']:
                output.append("**Episode Complete**")
            output.append("")

        # Summary
        scores = [float(s['reward']['score']) for s in steps]
        avg_score = sum(scores) / len(scores) if scores else 0
        output.append("---")
        output.append(f"**Average Score:** {avg_score:.2f}")
        output.append(f"**Total Steps:** {len(steps)}")

        return "\n".join(output)

    except Exception as e:
        return f"Error: {str(e)}"


def compare_modes() -> str:
    """Run comparison between LLM-rewarded and heuristic modes."""
    output = []
    output.append("## Mode Comparison")
    output.append("")

    for task in ["easy", "medium", "hard"]:
        output.append(f"### Task: {task.upper()}")
        output.append("")

        output.append("**LLM-Enabled Mode:**")
        try:
            steps_llm, summary_llm = run_episode(task, use_llm=True)
            scores_llm = [float(s['reward']['score']) for s in steps_llm]
            avg_llm = sum(scores_llm) / len(scores_llm) if scores_llm else 0
            output.append(f"- Action Policy: {_action_mode_label(summary_llm['action_mode'])}")
            output.append(f"- Reward Grading: {_reward_mode_label(summary_llm['reward_mode'])}")
            output.append(f"- Average Score: {avg_llm:.2f}")
            output.append(f"- Steps: {len(steps_llm)}")
        except Exception as e:
            output.append(f"- Error: {str(e)}")

        output.append("")

        output.append("**Heuristic Mode:**")
        try:
            steps_heur, summary_heur = run_episode(task, use_llm=False)
            scores_heur = [float(s['reward']['score']) for s in steps_heur]
            avg_heur = sum(scores_heur) / len(scores_heur) if scores_heur else 0
            output.append(f"- Action Policy: {_action_mode_label(summary_heur['action_mode'])}")
            output.append(f"- Reward Grading: {_reward_mode_label(summary_heur['reward_mode'])}")
            output.append(f"- Average Score: {avg_heur:.2f}")
            output.append(f"- Steps: {len(steps_heur)}")
        except Exception as e:
            output.append(f"- Error: {str(e)}")

        output.append("")

    output.append("---")
    output.append("*Note: LLM-enabled mode requires HF_TOKEN to be configured. The app reports action generation and reward grading separately so fallback is explicit.*")

    return "\n".join(output)


# Create Gradio Interface
with gr.Blocks(title="PriorityMind-Lite") as demo:
    gr.Markdown("""
    # PriorityMind-Lite
    ### LLM-Rewarded Customer Support Ticket Triage Environment
    
    **Meta PyTorch OpenEnv Hackathon 2026 | Team Axiom (IIT Madras)**
    
    This demo showcases an AI agent that learns to triage customer support tickets
    using rewards evaluated by Llama — not hardcoded rules.
    """)

    with gr.Tab("Interactive Demo"):
        with gr.Row():
            with gr.Column():
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="hard",
                    label="Select Task Difficulty"
                )
                llm_toggle = gr.Checkbox(
                    value=True,
                    label="Use LLM for actions and grading (requires HF_TOKEN)"
                )
                run_button = gr.Button("Run Demo", variant="primary")

            with gr.Column():
                output_markdown = gr.Markdown()

        run_button.click(
            fn=run_demo,
            inputs=[task_dropdown, llm_toggle],
            outputs=output_markdown
        )

    with gr.Tab("Mode Comparison"):
        compare_button = gr.Button("Compare enabled vs heuristic", variant="secondary")
        comparison_output = gr.Markdown()

        compare_button.click(
            fn=compare_modes,
            inputs=[],
            outputs=comparison_output
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ### How It Works
        
        1. **Environment Reset**: Generate a customer support ticket with sentiment and true category
        2. **Agent Takes Action**: Choose from categorize, prioritize, respond, escalate, or resolve
        3. **Hybrid Grader Evaluates**:
           - LLM Evaluation (60%): Llama judges on empathy, efficiency, strategy
           - Fallback (40%): Deterministic rules ensure reliability
        4. **Reward Signal**: Normalized score [0.0, 1.0] with partial signals
        
        ### Key Innovation
        
        Instead of defining "good customer service" mathematically, we let a language model
        (Llama) judge each agent action on dimensions like empathy, efficiency, and strategy.
        
        ### Technology Stack
        
        - **Framework**: OpenEnv (Meta)
        - **Model**: Llama 3.1 8B Instruct (via HF Router)
        - **Language**: Python 3.10
        - **Validation**: Pydantic typed models
        
        ### Links
        
        - [GitHub Repository](https://github.com/24f2006874/priority-mind-lite)
        - [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
        """)

    gr.Markdown("""
    ---
    *Built on OpenEnv. Judged by Llama.*
    """)


if __name__ == "__main__":
    configure_runtime()
    demo.launch(
        server_name=os.getenv("SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        theme=APP_THEME,
    )
