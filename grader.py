from __future__ import annotations

import json
import os
from typing import Any

from models import Action, Observation, Reward
from utils import clamp, normalize_text, contains_any

# Handle OpenAI import gracefully for when library is not installed
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Will raise RuntimeError if LLM evaluation is attempted


# Backward compatibility aliases - prefer utils module for new code
_clamp = clamp
_normalized_text = normalize_text
_contains_any = contains_any


def _efficiency_score(content: str | None) -> float:
    """Score efficiency on 0-10 scale, lower is better for conciseness."""
    text = _normalized_text(content)
    if not text:
        return 5.0  # Neutral score for non-text actions
    words = len(text.split())
    # Optimal response: 15-30 words
    if words <= 15:
        return 9.0  # Very concise
    if words <= 30:
        return 8.0  # Good length
    if words <= 50:
        return 6.5  # Acceptable
    if words <= 80:
        return 4.5  # Too verbose
    return 3.0  # Way too long


def _empathy_score(content: str | None) -> float:
    text = _normalized_text(content)
    if not text:
        return 3.0

    empathy_keywords = {
        "sorry",
        "apologize",
        "understand",
        "frustrated",
        "thanks",
        "thank you",
        "help",
        "urgent",
    }
    return 8.5 if _contains_any(text, empathy_keywords) else 4.0


class HybridGrader:
    """Combines deterministic rules with optional LLM judgement."""

    MAX_CACHE_SIZE = 1000  # Limit cache size to prevent memory issues

    def __init__(self, temperature: float = 0.0, timeout_seconds: float = 12.0, enable_llm: bool = True):
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.enable_llm = enable_llm
        self.cache: dict[str, Reward] = {}
        self.fallback = ProgrammaticGrader()

    def evaluate(
        self,
        obs: Observation,
        action: Action,
        task: str,
        ticket_context: dict[str, Any],
        last_action_error: str | None = None,
    ) -> Reward:
        programmatic_reward = self.fallback.evaluate(
            obs=obs,
            action=action,
            task=task,
            ticket_context=ticket_context,
            last_action_error=last_action_error,
        )

        cache_payload = {
            "task": task,
            "ticket_text": obs.ticket_text,
            "sentiment": obs.sentiment,
            "category": obs.category,
            "priority": obs.priority,
            "attempts": obs.attempts,
            "resolved": obs.resolved,
            "action_type": action.action_type,
            "action_content": action.content,
            "action_priority": action.priority,
            "true_category": ticket_context.get("true_category"),
            "last_action_error": last_action_error,
            "enable_llm": self.enable_llm,
        }
        cache_key = json.dumps(cache_payload, sort_keys=True)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.enable_llm:
            fallback_signals = dict(programmatic_reward.partial_signals)
            fallback_signals["used_fallback"] = True
            fallback_signals["fallback_reason"] = "llm_disabled"
            reward = Reward(
                score=programmatic_reward.score,
                reasoning=f"{programmatic_reward.reasoning} | llm_unavailable=llm_disabled",
                partial_signals=fallback_signals,
            )
        else:
            try:
                llm_reward = self._llm_evaluate(
                    obs=obs,
                    action=action,
                    task=task,
                    ticket_context=ticket_context,
                    last_action_error=last_action_error,
                )
                hybrid_score = _clamp(0.4 * programmatic_reward.score + 0.6 * llm_reward.score)
                partial_signals = {
                    **programmatic_reward.partial_signals,
                    **llm_reward.partial_signals,
                    "used_fallback": False,
                }
                reward = Reward(
                    score=hybrid_score,
                    reasoning=f"{llm_reward.reasoning} | fallback={programmatic_reward.reasoning}",
                    partial_signals=partial_signals,
                )
            except Exception as exc:
                fallback_signals = dict(programmatic_reward.partial_signals)
                fallback_signals["used_fallback"] = True
                fallback_signals["fallback_reason"] = str(exc)
                reward = Reward(
                    score=programmatic_reward.score,
                    reasoning=f"{programmatic_reward.reasoning} | llm_unavailable={exc}",
                    partial_signals=fallback_signals,
                )

        # Limit cache size to prevent memory issues
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = reward
        return reward

    def _llm_evaluate(
        self,
        obs: Observation,
        action: Action,
        task: str,
        ticket_context: dict[str, Any],
        last_action_error: str | None = None,
    ) -> Reward:
        if OpenAI is None:
            raise RuntimeError("openai library not installed")

        api_key = os.getenv("HF_TOKEN", "").strip()
        if not api_key:
            raise RuntimeError("missing_hf_token")

        # Build a more detailed prompt that emphasizes error handling
        error_guidance = ""
        if last_action_error:
            error_guidance = f"""
IMPORTANT: The action resulted in an error: "{last_action_error}"
This means the action was INVALID or INCOMPLETE.
Actions with errors should receive a LOW score (0-3 out of 10).
"""

        prompt = f"""You are evaluating an AI customer support triage agent.

TASK: {task}
GUIDANCE: {ticket_context.get("guidance", "")}
TRUE_CATEGORY: {ticket_context.get("true_category", "")}
EXPECTED_PRIORITY: {ticket_context.get("expected_priority", "")}
{error_guidance}
OBSERVATION:
- ticket_text: {obs.ticket_text}
- sentiment: {obs.sentiment:.2f}
- category: {obs.category}
- priority: {obs.priority}
- attempts: {obs.attempts}
- resolved: {str(obs.resolved).lower()}

ACTION:
- action_type: {action.action_type}
- content: {action.content}
- priority: {action.priority}
- last_action_error: {last_action_error}

Score the action for effectiveness, empathy, efficiency, and strategy.
IMPORTANT: If last_action_error is not "null" or None, the action FAILED and should score very low (0-3).
Return JSON only with keys:
score, reasoning, empathy, efficiency, strategy
Where score is 0-10 and sub-scores are 0-10.
"""

        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=api_key,
            timeout=float(os.getenv("HF_TIMEOUT_SECONDS", str(self.timeout_seconds))),
        )
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        payload = response.choices[0].message.content
        if not payload:
            raise RuntimeError("empty_llm_response")

        try:
            result = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_llm_json") from exc

        # Parse and validate score with proper error handling
        try:
            raw_score = result.get("score", 0.0)
            score = _clamp(float(raw_score) / 10.0)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"invalid_llm_score: {raw_score}") from exc

        reasoning = str(result.get("reasoning", "llm_evaluation"))

        # Normalize sub-scores from 0-10 to 0-1 range for consistency
        try:
            empathy = _clamp(float(result.get("empathy", 0.0)) / 10.0)
            efficiency = _clamp(float(result.get("efficiency", 0.0)) / 10.0)
            strategy = _clamp(float(result.get("strategy", 0.0)) / 10.0)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"invalid_llm_subscore: {exc}") from exc

        return Reward(
            score=score,
            reasoning=reasoning,
            partial_signals={
                "empathy": empathy,
                "efficiency": efficiency,
                "strategy": strategy,
            },
        )


class ProgrammaticGrader:
    """Deterministic task-aware fallback grader."""

    def evaluate(
        self,
        obs: Observation,
        action: Action,
        task: str,
        ticket_context: dict[str, Any],
        last_action_error: str | None = None,
    ) -> Reward:
        content = _normalized_text(action.content)
        true_category = ticket_context.get("true_category")
        expected_priority = ticket_context.get("expected_priority")
        escalated = bool(ticket_context.get("escalated", False))

        empathy = _empathy_score(action.content if action.action_type == "respond" else None)
        efficiency = _efficiency_score(action.content if action.action_type == "respond" else None)
        strategy = 3.5
        score = 0.0  # Start neutral, no positive bias
        notes: list[str] = []

        # Apply significant penalty for action errors
        if last_action_error:
            score -= 0.30
            strategy = 1.0
            notes.append(f"invalid_action={last_action_error}")

        if task == "easy":
            if action.action_type == "categorize":
                if content == true_category:
                    score += 0.45
                    strategy = 8.5
                    notes.append("correct_billing_category")
                else:
                    score -= 0.05
                    strategy = 2.0
                    notes.append("incorrect_category")
            elif action.action_type == "prioritize":
                if action.priority is None:
                    score -= 0.20
                    strategy = 1.0
                    notes.append("missing_priority_value")
                elif action.priority == expected_priority:
                    score += 0.25
                    strategy = 7.5
                    notes.append("reasonable_priority")
                elif action.priority in {"high", "urgent"}:
                    score += 0.10
                    strategy = 5.5
                    notes.append("aggressive_priority")
            elif action.action_type == "respond":
                if _contains_any(content, {"bill", "billing", "charge", "review"}):
                    score += 0.15
                    strategy = 6.5
                    notes.append("helpful_billing_response")
            elif action.action_type == "resolve":
                if obs.category == true_category and obs.attempts >= 2:
                    score += 0.20
                    strategy = 7.0
                    notes.append("resolved_after_basic_triage")
                else:
                    score -= 0.10
                    strategy = 2.5
                    notes.append("premature_resolution")

        elif task == "medium":
            if action.action_type == "categorize":
                if content == true_category:
                    score += 0.30
                    strategy = 7.5
                    notes.append("correct_technical_category")
                else:
                    score -= 0.08
                    strategy = 2.0
                    notes.append("wrong_category")
            elif action.action_type == "prioritize":
                if action.priority is None:
                    score -= 0.20
                    strategy = 1.0
                    notes.append("missing_priority_value")
                elif action.priority in {"high", "urgent"}:
                    score += 0.30
                    strategy = 8.0
                    notes.append("urgent_technical_priority")
                else:
                    score -= 0.05
                    strategy = 3.0
                    notes.append("under_prioritized")
            elif action.action_type == "respond":
                if empathy >= 8.0:
                    score += 0.15
                    notes.append("empathetic_tone")
                if _contains_any(content, {"crash", "restart", "device", "update", "log"}):
                    score += 0.10
                    strategy = 7.0
                    notes.append("technical_guidance")
            elif action.action_type == "resolve":
                if obs.attempts < ticket_context.get("recommended_resolution_after", 3):
                    score -= 0.12
                    strategy = 2.0
                    notes.append("resolved_too_early")
                else:
                    score += 0.10
                    strategy = 6.0
                    notes.append("resolved_after_guidance")

        elif task == "hard":
            if action.action_type == "categorize":
                if content == true_category:
                    score += 0.25
                    strategy = 7.0
                    notes.append("correct_complaint_category")
                else:
                    score -= 0.10
                    strategy = 2.0
                    notes.append("wrong_complaint_category")
            elif action.action_type == "prioritize":
                if action.priority is None:
                    score -= 0.20
                    strategy = 1.0
                    notes.append("missing_priority_value")
                elif action.priority in {"urgent", "high"}:
                    score += 0.15
                    strategy = 6.5
                    notes.append("urgent_priority")
                else:
                    score -= 0.08
                    strategy = 2.5
                    notes.append("under_prioritized")
            elif action.action_type == "respond":
                if empathy >= 8.0:
                    score += 0.12
                    notes.append("empathetic_response")
                if _contains_any(content, {"refund", "data", "escalate", "specialist", "investigate"}):
                    score += 0.08
                    strategy = 6.5
                    notes.append("multi_issue_acknowledged")
            elif action.action_type == "escalate":
                score += 0.30
                strategy = 9.0
                notes.append("strategic_escalation")
            elif action.action_type == "resolve":
                if not escalated and obs.attempts < ticket_context.get("recommended_resolution_after", 4):
                    score -= 0.20
                    strategy = 1.5
                    notes.append("premature_hard_resolution")
                else:
                    score += 0.10
                    strategy = 6.0
                    notes.append("resolved_after_escalation")

        final_score = _clamp(score)
        reasoning = ", ".join(notes) if notes else "limited_progress"
        return Reward(
            score=final_score,
            reasoning=reasoning,
            partial_signals={
                "empathy": round(_clamp(empathy / 10.0), 2),
                "efficiency": round(_clamp(efficiency / 10.0), 2),
                "strategy": round(_clamp(strategy / 10.0), 2),
            },
        )
