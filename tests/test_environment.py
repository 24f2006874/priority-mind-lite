from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from environment import PriorityMindEnv, TASK_VARIATIONS
from grader import HybridGrader, ProgrammaticGrader
from inference import evaluate_success, mock_action
from models import Action, Observation, Reward
from utils import contains_any, infer_ticket_category

ROOT = Path(__file__).resolve().parents[1]


def test_environment_creation_for_all_tasks():
    assert PriorityMindEnv("easy").max_steps == 3
    assert PriorityMindEnv("medium").max_steps == 5
    assert PriorityMindEnv("hard").max_steps == 8


def test_reset_is_deterministic_with_fixed_variation():
    # Use variation_index=0 for deterministic testing
    env = PriorityMindEnv("easy", variation_index=0)
    first = env.reset()
    second = env.reset()
    assert first.model_dump() == second.model_dump()


def test_state_returns_serializable_dict():
    # Use variation_index=0 to get the first variation
    env = PriorityMindEnv("medium", variation_index=0)
    env.reset()
    state = env.state()
    assert isinstance(state, dict)
    # With variation_index=0, we get the first variation
    assert state["ticket_text"] == "App keeps crashing! I'm so frustrated!"


def test_step_updates_category_and_attempt_count():
    env = PriorityMindEnv("easy")
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="categorize", content="billing"))
    assert obs.category == "billing"
    assert obs.attempts == 1
    assert isinstance(reward, Reward)
    assert done is False
    assert info["last_action_error"] is None


def test_prioritize_requires_priority_value():
    # Use enable_llm=False to test the programmatic grader behavior
    env = PriorityMindEnv("easy", enable_llm=False)
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="prioritize"))
    assert obs.priority is None
    assert reward.score <= 0.1
    assert info["last_action_error"] == "missing_priority"
    assert done is False


def test_resolve_terminates_episode():
    env = PriorityMindEnv("easy")
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="resolve"))
    assert obs.resolved is True
    assert done is True
    assert info["step"] == 1


def test_max_step_termination_occurs():
    env = PriorityMindEnv("easy")
    env.reset()
    done = False
    for _ in range(env.max_steps):
        _, _, done, _ = env.step(Action(action_type="respond", content="We are reviewing your bill."))
    assert done is True


def test_models_remain_strict_and_typed():
    obs = Observation(ticket_text="x", sentiment=-0.4)
    action = Action(action_type="categorize", content="billing")
    reward = Reward(score=0.5, reasoning="ok", partial_signals={"empathy": 5})
    assert obs.ticket_text == "x"
    assert action.action_type == "categorize"
    assert reward.score == 0.5


def test_programmatic_grader_rewards_correct_easy_category_more_than_wrong_one():
    grader = ProgrammaticGrader()
    context = {"true_category": "billing", "expected_priority": "medium", "recommended_resolution_after": 2}
    obs = Observation(ticket_text="My bill is higher than expected", sentiment=-0.3)
    good = grader.evaluate(obs, Action(action_type="categorize", content="billing"), "easy", context)
    bad = grader.evaluate(obs, Action(action_type="categorize", content="technical"), "easy", context)
    assert good.score > bad.score


def test_programmatic_grader_rewards_medium_priority_alignment():
    grader = ProgrammaticGrader()
    context = {"true_category": "technical", "expected_priority": "high", "recommended_resolution_after": 3}
    obs = Observation(ticket_text="App keeps crashing! I'm so frustrated!", sentiment=-0.8)
    high = grader.evaluate(obs, Action(action_type="prioritize", priority="high"), "medium", context)
    low = grader.evaluate(obs, Action(action_type="prioritize", priority="low"), "medium", context)
    assert high.score > low.score


def test_programmatic_grader_discourages_premature_hard_resolution():
    grader = ProgrammaticGrader()
    context = {
        "true_category": "complaint",
        "expected_priority": "urgent",
        "recommended_resolution_after": 4,
        "escalated": False,
    }
    obs = Observation(
        ticket_text="I've waited 3 days for a refund AND your app deleted my data. This is unacceptable!",
        sentiment=-0.95,
        attempts=1,
    )
    reward = grader.evaluate(obs, Action(action_type="resolve"), "hard", context)
    assert reward.score < 0.1


def test_hybrid_grader_falls_back_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    grader = HybridGrader()
    obs = Observation(ticket_text="Test", sentiment=-0.2)
    action = Action(action_type="respond", content="I am sorry. I will help.")
    context = {"true_category": "general", "expected_priority": "low", "recommended_resolution_after": 2}
    reward = grader.evaluate(obs, action, "easy", context)
    assert 0.0 <= reward.score <= 1.0
    assert reward.partial_signals["used_fallback"] is True


def test_cache_keys_differentiate_action_content(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    grader = HybridGrader()
    obs = Observation(ticket_text="Test", sentiment=-0.2, attempts=2)
    context = {"true_category": "technical", "expected_priority": "high", "recommended_resolution_after": 3}
    reward_one = grader.evaluate(obs, Action(action_type="respond", content="I am sorry and I will investigate."), "medium", context)
    reward_two = grader.evaluate(obs, Action(action_type="respond", content="Restart the app and send logs."), "medium", context)
    assert len(grader.cache) == 2
    assert reward_one.reasoning != ""
    assert reward_two.reasoning != ""


def test_inference_mock_output_is_strict():
    completed = subprocess.run(
        [sys.executable, "inference.py", "--mock"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines
    assert all(line.startswith(("[START]", "[STEP]", "[END]")) for line in lines)
    assert sum(line.startswith("[START]") for line in lines) == 3
    assert sum(line.startswith("[END]") for line in lines) == 3
    assert any("error=null" in line for line in lines if line.startswith("[STEP]"))


def test_inference_mock_output_uses_lowercase_booleans():
    completed = subprocess.run(
        [sys.executable, "inference.py", "--mock"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    step_lines = [line for line in completed.stdout.splitlines() if line.startswith("[STEP]")]
    end_lines = [line for line in completed.stdout.splitlines() if line.startswith("[END]")]
    assert all("done=true" in line or "done=false" in line for line in step_lines)
    assert all("success=true" in line or "success=false" in line for line in end_lines)


def test_inference_verbose_reports_mode():
    completed = subprocess.run(
        [sys.executable, "inference.py", "--mock", "--task", "easy", "--verbose"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert any(line == "summary mode=mock" for line in lines)
    assert any(line.startswith("summary task=easy score=") and "mode=mock" in line for line in lines)


def test_hybrid_grader_respects_explicit_mock_mode(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    grader = HybridGrader(enable_llm=False)
    obs = Observation(ticket_text="Test", sentiment=-0.2)
    action = Action(action_type="respond", content="I am sorry. I will help.")
    context = {"true_category": "general", "expected_priority": "low", "recommended_resolution_after": 2}
    reward = grader.evaluate(obs, action, "easy", context)
    assert reward.partial_signals["used_fallback"] is True
    assert reward.partial_signals["fallback_reason"] == "llm_disabled"


def test_mock_policy_solves_all_bundled_variations():
    for task, variations in TASK_VARIATIONS.items():
        for index in range(len(variations)):
            env = PriorityMindEnv(task=task, variation_index=index, enable_llm=False)
            obs = env.reset()
            actions: list[Action] = []
            rewards: list[float] = []
            done = False

            while not done:
                action = mock_action(obs, task)
                actions.append(action)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward.score)

            final_score = sum(rewards) / len(rewards) if rewards else 0.0
            assert evaluate_success(task, obs, actions, final_score), (task, index, obs.ticket_text)


def test_contains_any_uses_word_boundaries_for_single_tokens():
    assert contains_any("the app crashes", {"app"}) is True
    assert contains_any("this happened yesterday", {"app"}) is False


def test_infer_ticket_category_avoids_substring_false_positives():
    assert infer_ticket_category("I am happy with support") == "general"
    assert infer_ticket_category("This happened yesterday") == "general"
    assert infer_ticket_category("Application keeps crashing") == "technical"


def test_programmatic_partial_signals_are_normalized_to_unit_interval():
    grader = ProgrammaticGrader()
    context = {"true_category": "technical", "expected_priority": "high", "recommended_resolution_after": 3}
    obs = Observation(ticket_text="App keeps crashing!", sentiment=-0.8)
    reward = grader.evaluate(
        obs,
        Action(action_type="respond", content="I am sorry. Please restart and share logs so we can help quickly."),
        "medium",
        context,
    )
    for key in ("empathy", "efficiency", "strategy"):
        value = reward.partial_signals[key]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
