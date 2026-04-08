from __future__ import annotations

import random
from typing import Any, Dict, Literal, Optional

from grader import HybridGrader
from models import Action, Observation, Reward

TaskName = Literal["easy", "medium", "hard"]

# Task variations for real-world diversity
# Each difficulty level has multiple ticket variations to simulate real customer support scenarios
TASK_VARIATIONS: dict[TaskName, list[dict[str, Any]]] = {
    "easy": [
        {
            "ticket_text": "My bill is higher than expected",
            "sentiment": -0.3,
            "guidance": "Categorize the issue as billing and set a sensible priority.",
        },
        {
            "ticket_text": "I was charged twice for my subscription",
            "sentiment": -0.4,
            "guidance": "Categorize the issue as billing and set a sensible priority.",
        },
        {
            "ticket_text": "Can you explain the charges on my latest invoice?",
            "sentiment": -0.1,
            "guidance": "Categorize the issue as billing and set a sensible priority.",
        },
        {
            "ticket_text": "My payment failed but I was still charged",
            "sentiment": -0.5,
            "guidance": "Categorize the issue as billing and set a sensible priority.",
        },
        {
            "ticket_text": "I need a refund for the overcharge on my account",
            "sentiment": -0.35,
            "guidance": "Categorize the issue as billing and set a sensible priority.",
        },
        {
            "ticket_text": "Why did my monthly fee increase without notice?",
            "sentiment": -0.45,
            "guidance": "Categorize the issue as billing and set a sensible priority.",
        },
    ],
    "medium": [
        {
            "ticket_text": "App keeps crashing! I'm so frustrated!",
            "sentiment": -0.8,
            "guidance": "Acknowledge frustration, route correctly, and offer a helpful next step.",
        },
        {
            "ticket_text": "The app freezes every time I try to upload a photo",
            "sentiment": -0.7,
            "guidance": "Acknowledge frustration, route correctly, and offer a helpful next step.",
        },
        {
            "ticket_text": "Login page shows error 500 constantly",
            "sentiment": -0.75,
            "guidance": "Acknowledge frustration, route correctly, and offer a helpful next step.",
        },
        {
            "ticket_text": "My notifications stopped working after the update",
            "sentiment": -0.65,
            "guidance": "Acknowledge frustration, route correctly, and offer a helpful next step.",
        },
        {
            "ticket_text": "The search function returns no results even for items I know exist",
            "sentiment": -0.72,
            "guidance": "Acknowledge frustration, route correctly, and offer a helpful next step.",
        },
        {
            "ticket_text": "App drains my battery in just 2 hours",
            "sentiment": -0.68,
            "guidance": "Acknowledge frustration, route correctly, and offer a helpful next step.",
        },
    ],
    "hard": [
        {
            "ticket_text": "I've waited 3 days for a refund AND your app deleted my data. This is unacceptable!",
            "sentiment": -0.95,
            "guidance": "Handle the complaint with empathy, urgency, and strategic escalation.",
        },
        {
            "ticket_text": "Your service has been down for 2 days and I'm losing business. I want compensation!",
            "sentiment": -0.92,
            "guidance": "Handle the complaint with empathy, urgency, and strategic escalation.",
        },
        {
            "ticket_text": "I've been transferred 5 times and no one has solved my problem. This is terrible service!",
            "sentiment": -0.9,
            "guidance": "Handle the complaint with empathy, urgency, and strategic escalation.",
        },
        {
            "ticket_text": "My account was hacked and your support team is not responding fast enough!",
            "sentiment": -0.93,
            "guidance": "Handle the complaint with empathy, urgency, and strategic escalation.",
        },
        {
            "ticket_text": "You charged me for a year subscription but I only wanted monthly. Refund the difference NOW!",
            "sentiment": -0.88,
            "guidance": "Handle the complaint with empathy, urgency, and strategic escalation.",
        },
        {
            "ticket_text": "My personal data was exposed in your data breach and I haven't heard from you in a week!",
            "sentiment": -0.97,
            "guidance": "Handle the complaint with empathy, urgency, and strategic escalation.",
        },
    ],
}

# Task-level configuration (consistent across variations)
TASK_META: dict[TaskName, dict[str, Any]] = {
    "easy": {
        "true_category": "billing",
        "expected_priority": "medium",
        "max_steps": 3,
        "recommended_resolution_after": 2,
    },
    "medium": {
        "true_category": "technical",
        "expected_priority": "high",
        "max_steps": 5,
        "recommended_resolution_after": 3,
    },
    "hard": {
        "true_category": "complaint",
        "expected_priority": "urgent",
        "max_steps": 8,
        "recommended_resolution_after": 4,
    },
}

VALID_CATEGORIES = {"billing", "technical", "general", "complaint"}


class PriorityMindEnv:
    """Customer-support triage environment with randomized task variations.
    
    Each difficulty level has multiple ticket variations to simulate real-world
    customer support scenarios. The environment randomly selects a variation
    on each reset, ensuring diverse training and evaluation.
    """

    def __init__(
        self,
        task: TaskName = "easy",
        seed: Optional[int] = None,
        variation_index: Optional[int] = None,
        enable_llm: bool = True,
    ):
        if task not in TASK_VARIATIONS:
            raise ValueError(f"Unsupported task: {task}")

        self.task: TaskName = task
        self.grader = HybridGrader(enable_llm=enable_llm)
        self.max_steps = TASK_META[task]["max_steps"]
        self.step_count = 0
        self._observation: Observation | None = None
        self._ticket_context: dict[str, Any] = {}
        self._last_action_error: str | None = None
        self._variation_index = variation_index  # Fixed variation for deterministic testing
        self._rng = random.Random(seed)  # Deterministic randomization with optional seed
        self._reset_state()

    def _reset_state(self) -> None:
        # Select a variation: fixed index, or random from RNG
        if self._variation_index is not None:
            variation = TASK_VARIATIONS[self.task][self._variation_index % len(TASK_VARIATIONS[self.task])]
        else:
            variation = self._rng.choice(TASK_VARIATIONS[self.task])
        meta = TASK_META[self.task]
        
        self._ticket_context = {
            "task": self.task,
            "ticket_text": variation["ticket_text"],
            "true_category": meta["true_category"],
            "expected_priority": meta["expected_priority"],
            "recommended_resolution_after": meta["recommended_resolution_after"],
            "guidance": variation["guidance"],
        }
        self._observation = Observation(
            ticket_text=variation["ticket_text"],
            sentiment=variation["sentiment"],
            category=None,
            priority=None,
            attempts=0,
            resolved=False,
        )
        self.step_count = 0
        self._last_action_error = None

    def reset(self) -> Observation:
        self._reset_state()
        return self._require_observation()

    def _require_observation(self) -> Observation:
        if self._observation is None:
            raise RuntimeError("Environment has not been initialized correctly.")
        return self._observation

    def _apply_action(self, action: Action) -> None:
        observation = self._require_observation()
        self._last_action_error = None

        if action.action_type == "categorize":
            if action.content not in VALID_CATEGORIES:
                self._last_action_error = "invalid_category"
                return
            observation.category = action.content
            return

        if action.action_type == "prioritize":
            if action.priority is None:
                self._last_action_error = "missing_priority"
                return
            observation.priority = action.priority
            return

        if action.action_type == "respond":
            if not action.content:
                self._last_action_error = "missing_response_content"
                return
            return

        if action.action_type == "escalate":
            self._ticket_context["escalated"] = True
            return

        if action.action_type == "resolve":
            observation.resolved = True
            return

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        observation = self._require_observation()
        self.step_count += 1
        self._apply_action(action)
        observation.attempts += 1

        reward = self.grader.evaluate(
            obs=observation,
            action=action,
            task=self.task,
            ticket_context=self._ticket_context,
            last_action_error=self._last_action_error,
        )

        done = observation.resolved or self.step_count >= self.max_steps
        info = {
            "task": self.task,
            "step": self.step_count,
            "last_action_error": self._last_action_error,
            "reasoning": reward.reasoning,
            "used_fallback": bool(reward.partial_signals.get("used_fallback", False)),
        }
        return observation, reward, done, info

    def state(self) -> dict[str, Any]:
        return self._require_observation().model_dump()

    def get_state(self) -> dict[str, Any]:
        """Backward-compatible alias for older callers."""
        return self.state()

    def close(self) -> None:
        self._observation = None

    @property
    def num_variations(self) -> dict[TaskName, int]:
        """Return the number of variations available for each task."""
        return {task: len(variations) for task, variations in TASK_VARIATIONS.items()}
