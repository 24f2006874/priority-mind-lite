"""
PriorityMind-Lite: Gradio Web Interface for Hugging Face Spaces
================================================================
This provides an interactive demo of the PriorityMind-Lite environment
where users can interact with the AI customer support triage agent.

Deployed at: https://huggingface.co/spaces/raunakratan/priority-mind-lite
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
    """Run demo and return formatted HTML results."""
    try:
        steps, summary = run_episode(task, use_llm)

        output = []
        output.append(f"<h1>PriorityMind-Lite Demo</h1>")
        output.append(f"<h2>Task: {task.upper()}</h2>")
        output.append(f"<p><strong>Action Policy:</strong> {_action_mode_label(summary['action_mode'])}</p>")
        output.append(f"<p><strong>Reward Grading:</strong> {_reward_mode_label(summary['reward_mode'])}</p>")
        
        if use_llm and not summary["client_available"]:
            output.append("<div class='alert alert-warning'>[!] Note: HF_TOKEN or OpenAI client unavailable, using heuristic actions and programmatic grading only.</div>")
        elif summary["action_mode"] == "mixed":
            output.append(f"<div class='alert alert-info'>[i] Action generation used LLM in {summary['llm_action_steps']}/{summary['total_steps']} steps, heuristic fallback for the rest.</div>")
        elif summary["action_mode"] == "heuristic_fallback":
            output.append("<div class='alert alert-warning'>[!] Every action fell back to heuristic policy after LLM failure.</div>")

        if summary["reward_mode"] == "hybrid_mixed":
            output.append(f"<div class='alert alert-info'>[i] Reward grading used LLM judge in {summary['reward_llm_steps']}/{summary['total_steps']} steps, programmatic fallback in the rest.</div>")
        elif summary["reward_mode"] == "programmatic_fallback":
            output.append("<div class='alert alert-warning'>[!] Reward grading attempted live evaluation, but every step fell back to programmatic grader.</div>")

        output.append("<div class='steps-container'>")

        for step_info in steps:
            output.append(f"<div class='step-card'>")
            output.append(f"<h3>Step {step_info['step']}</h3>")
            output.append(f"<p><strong>Action:</strong> {step_info['action']} <span class='badge badge-secondary'>({step_info['action_source']})</span></p>")
            output.append(f"<p><strong>Reward Source:</strong> {step_info['reward_source']}</p>")
            output.append(f"<p><strong>Reward Score:</strong> <span class='score'>{step_info['reward']['score']}</span></p>")
            output.append(f"<p><strong>Reasoning:</strong> {step_info['reward']['reasoning']}</p>")
            
            # Progress bars for partial signals
            empathy_val = float(step_info['partial_signals']['empathy'].split('/')[0])
            efficiency_val = float(step_info['partial_signals']['efficiency'].split('/')[0])
            strategy_val = float(step_info['partial_signals']['strategy'].split('/')[0])
            
            def get_color(val):
                if val >= 0.8:
                    return "#28a745"  # green
                elif val >= 0.6:
                    return "#ffc107"  # yellow
                else:
                    return "#dc3545"  # red
            
            output.append("<div class='signals'>")
            output.append(f"<div class='signal'><span class='icon'>EMPATHY</span> <div class='progress-bar'><div class='progress-fill' style='width: {empathy_val*100}%; background-color: {get_color(empathy_val)}'></div></div> <span class='signal-value'>{step_info['partial_signals']['empathy']}</span></div>")
            output.append(f"<div class='signal'><span class='icon'>EFFICIENCY</span> <div class='progress-bar'><div class='progress-fill' style='width: {efficiency_val*100}%; background-color: {get_color(efficiency_val)}'></div></div> <span class='signal-value'>{step_info['partial_signals']['efficiency']}</span></div>")
            output.append(f"<div class='signal'><span class='icon'>STRATEGY</span> <div class='progress-bar'><div class='progress-fill' style='width: {strategy_val*100}%; background-color: {get_color(strategy_val)}'></div></div> <span class='signal-value'>{step_info['partial_signals']['strategy']}</span></div>")
            output.append("</div>")
            
            if step_info["action_error"]:
                output.append(f"<p class='error'>[ERROR] Action Error: {step_info['action_error']}</p>")
            if step_info['done']:
                output.append("<p class='success'>[OK] Episode Complete</p>")
            output.append("</div>")

        output.append("</div>")

        # Summary with score chart
        scores = [float(s['reward']['score']) for s in steps]
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Simple bar chart for scores
        chart_html = "<div class='score-chart'>"
        for i, score in enumerate(scores, 1):
            color = get_color(score)
            chart_html += f"<div class='chart-bar' style='height: {score*100}px; background-color: {color}' title='Step {i}: {score:.2f}'></div>"
        chart_html += "</div>"
        
        output.append("<div class='summary'>")
        output.append("---")
        output.append("<h3>Summary</h3>")
        output.append(f"<p><strong>Average Score:</strong> <span class='avg-score'>{avg_score:.2f}</span></p>")
        output.append(f"<p><strong>Min Score:</strong> {min_score:.2f} | <strong>Max Score:</strong> {max_score:.2f}</p>")
        output.append(f"<p><strong>Total Steps:</strong> {len(steps)}</p>")
        output.append("<h4>Score Progression</h4>")
        output.append(chart_html)
        output.append("</div>")

        # Add CSS
        css = """
        <style>
        body { color: #000; }
        .alert { padding: 12px 15px; margin: 10px 0; border-radius: 5px; font-weight: 500; }
        .alert-warning { background-color: #fff3cd; border: 2px solid #ffc107; color: #000; }
        .alert-info { background-color: #d1ecf1; border: 2px solid #17a2b8; color: #000; }
        .alert-danger { background-color: #f8d7da; border: 2px solid #dc3545; color: #000; }
        .step-card { border: 2px solid #ddd; border-radius: 8px; padding: 20px; margin: 15px 0; background-color: #fff; }
        .step-card h3 { color: #000; margin: 0 0 10px 0; }
        .step-card p { color: #000; margin: 8px 0; }
        .badge { display: inline-block; padding: 4px 8px; font-size: 0.85em; border-radius: 3px; background-color: #6c757d; color: white; }
        .badge-secondary { background-color: #6c757d; }
        .score { font-size: 1.2em; font-weight: bold; color: #0056b3; }
        .signals { margin-top: 15px; }
        .signal { display: flex; align-items: center; margin: 10px 0; }
        .signal .icon { width: 100px; font-weight: bold; font-size: 0.9em; color: #000; }
        .progress-bar { flex: 1; height: 15px; background-color: #e9ecef; border-radius: 5px; margin: 0 10px; overflow: hidden; border: 1px solid #ccc; }
        .progress-fill { height: 100%; transition: width 0.3s; }
        .signal-value { width: 80px; text-align: right; font-weight: bold; color: #000; }
        .error { color: #000; font-weight: bold; }
        .success { color: #000; font-weight: bold; }
        .summary { margin-top: 30px; padding: 25px; background-color: #fff; border: 3px solid #007bff; border-radius: 8px; }
        .summary h3 { color: #000; margin: 0 0 15px 0; font-size: 1.5em; }
        .summary h4 { color: #000; margin: 15px 0 10px 0; }
        .summary p { color: #000; font-size: 1.1em; margin: 10px 0; font-weight: 500; }
        .avg-score { font-size: 1.3em; font-weight: bold; color: #000; }
        .score-chart { display: flex; align-items: end; justify-content: space-around; height: 150px; margin: 15px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border: 2px solid #ddd; }
        .chart-bar { width: 25px; min-height: 10px; border-radius: 3px 3px 0 0; margin: 0 3px; }
        .steps-container { margin: 20px 0; }
        </style>
        """

        return css + "\n".join(output)

    except Exception as e:
        return f"<div class='alert alert-danger'>Error: {str(e)}</div>"


def compare_modes() -> str:
    """Run comparison between LLM-rewarded and heuristic modes."""
    output = []
    output.append("<h2>Mode Comparison</h2>")

    for task in ["easy", "medium", "hard"]:
        output.append(f"<h3>Task: {task.upper()}</h3>")

        output.append("<div class='mode-section'>")
        output.append("<h4>LLM-Enabled Mode</h4>")
        try:
            steps_llm, summary_llm = run_episode(task, use_llm=True)
            scores_llm = [float(s['reward']['score']) for s in steps_llm]
            avg_llm = sum(scores_llm) / len(scores_llm) if scores_llm else 0
            output.append(f"<p><strong>Action Policy:</strong> {_action_mode_label(summary_llm['action_mode'])}</p>")
            output.append(f"<p><strong>Reward Grading:</strong> {_reward_mode_label(summary_llm['reward_mode'])}</p>")
            output.append(f"<p><strong>Average Score:</strong> <span class='score'>{avg_llm:.2f}</span></p>")
            output.append(f"<p><strong>Steps:</strong> {len(steps_llm)}</p>")
        except Exception as e:
            output.append(f"<p class='error'>Error: {str(e)}</p>")
        output.append("</div>")

        output.append("<div class='mode-section'>")
        output.append("<h4>Heuristic Mode</h4>")
        try:
            steps_heur, summary_heur = run_episode(task, use_llm=False)
            scores_heur = [float(s['reward']['score']) for s in steps_heur]
            avg_heur = sum(scores_heur) / len(scores_heur) if scores_heur else 0
            output.append(f"<p><strong>Action Policy:</strong> {_action_mode_label(summary_heur['action_mode'])}</p>")
            output.append(f"<p><strong>Reward Grading:</strong> {_reward_mode_label(summary_heur['reward_mode'])}</p>")
            output.append(f"<p><strong>Average Score:</strong> <span class='score'>{avg_heur:.2f}</span></p>")
            output.append(f"<p><strong>Steps:</strong> {len(steps_heur)}</p>")
        except Exception as e:
            output.append(f"<p class='error'>Error: {str(e)}</p>")
        output.append("</div>")

    output.append("<hr>")
    output.append("<p><em>Note: LLM-enabled mode requires HF_TOKEN to be configured. The app reports action generation and reward grading separately so fallback is explicit.</em></p>")

    css = """
    <style>
    .mode-section { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f9f9f9; }
    .score { font-size: 1.2em; font-weight: bold; color: #000; }
    .error { color: #000; }
    </style>
    """

    return css + "\n".join(output)


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
                output_html = gr.HTML(label="Demo Results")

        run_button.click(
            fn=run_demo,
            inputs=[task_dropdown, llm_toggle],
            outputs=output_html
        )

    with gr.Tab("Mode Comparison"):
        compare_button = gr.Button("Compare enabled vs heuristic", variant="secondary")
        comparison_output = gr.HTML()

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
