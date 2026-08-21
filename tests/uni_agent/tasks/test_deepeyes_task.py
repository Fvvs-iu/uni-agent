from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import uni_agent.tasks.deepeyes.reward as reward_module
import uni_agent.tasks.deepeyes.task as task_module
from uni_agent.agents import AgentResult, ModelConfig
from uni_agent.agents.deepeyes import DeepEyesAgentConfig
from uni_agent.tasks import get_task
from uni_agent.tasks.deepeyes import DeepEyesTask, DeepEyesTaskConfig, compute_score


class _JudgeCompletions:
    def __init__(self, judgement: str):
        self.judgement = judgement
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.judgement))]
        )


class _SandboxContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAgent:
    async def run(self, *, sandbox, messages):
        return AgentResult(
            output={"final_answer": "<answer>cat</answer>", "termination_reason": "finished"},
            info={
                "termination_reason": "finished",
                "steps": 2,
                "tool_calls": 1,
                "tool_successes": 1,
                "tool_errors": 0,
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
            finished=True,
        )


def _reward_context(**overrides):
    context = {
        "question": "What animal is shown?",
        "finished": True,
        "final_answer": "<answer>cat</answer>",
        "tool_calls": 1,
        "tool_successes": 1,
        "tool_errors": 0,
    }
    context.update(overrides)
    return context


def test_deepeyes_task_is_registered():
    task = get_task(
        {
            "name": "deepeyes",
            "sandbox": {"provider": "local"},
            "agent": {
                "name": "deepeyes",
                "model": {"base_url": "http://gateway/v1", "model_name": "policy"},
            },
            "prompt": [{"role": "user", "content": "question"}],
            "question": "question",
            "ground_truth": "answer",
        }
    )

    assert isinstance(task, DeepEyesTask)
    assert isinstance(task.config, DeepEyesTaskConfig)
    assert isinstance(task.config.agent, DeepEyesAgentConfig)


def test_reward_grants_accuracy_and_successful_tool_bonus(monkeypatch):
    completions = _JudgeCompletions("CORRECT")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(reward_module, "_get_judge_client", lambda _config: (client, "judge"))

    score = compute_score("deepeyes", "<answer>cat</answer>", "cat", _reward_context())

    assert score["score"] == pytest.approx(2.0)
    assert score["acc"] == 1.0
    assert score["format"] == 0.0
    assert score["tool"] == 1.0
    assert len(completions.calls) == 1


def test_reward_requires_successful_tool_execution_for_bonus(monkeypatch):
    completions = _JudgeCompletions("CORRECT")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(reward_module, "_get_judge_client", lambda _config: (client, "judge"))

    score = compute_score(
        "deepeyes",
        "<answer>cat</answer>",
        "cat",
        _reward_context(tool_successes=0, tool_errors=1),
    )

    assert score["score"] == pytest.approx(0.8)
    assert score["tool"] == 0.0


def test_reward_skips_judge_for_unfinished_episode(monkeypatch):
    def fail_if_called(_config):
        raise AssertionError("judge must not be initialized")

    monkeypatch.setattr(reward_module, "_get_judge_client", fail_if_called)

    score = compute_score(
        "deepeyes",
        "<answer>cat</answer>",
        "cat",
        _reward_context(finished=False),
    )

    assert score["score"] == pytest.approx(-0.2)
    assert score["acc"] == 0.0
    assert score["finished"] == 0.0


def test_task_runs_agent_and_composes_result(monkeypatch):
    config = DeepEyesTaskConfig(
        sandbox={"provider": "local"},
        agent=DeepEyesAgentConfig(model=ModelConfig(base_url="http://gateway/v1", model_name="policy")),
        prompt=[{"role": "user", "content": "What animal is shown?"}],
        question="What animal is shown?",
        ground_truth="cat",
        data_source="deepeyes",
    )
    task = DeepEyesTask(config)
    monkeypatch.setattr(task, "build_sandbox", lambda: _SandboxContext())
    monkeypatch.setattr(task, "build_agent", lambda: _FakeAgent())

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(task_module.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(
        task_module,
        "compute_score",
        lambda *args, **kwargs: {
            "score": 2.0,
            "acc": 1.0,
            "format": 0.0,
            "tool": 1.0,
            "finished": 1.0,
            "tool_calls": 1.0,
            "tool_successes": 1.0,
            "tool_errors": 0.0,
            "answer_tags": 1.0,
        },
    )

    result = asyncio.run(task.run())

    assert result.reward == 2.0
    assert result.accuracy == 1.0
    assert result.finished is True
    assert result.extra_info["tool_successes"] == 1.0
    assert result.extra_info["steps"] == 2
    assert result.extra_info["total_tokens"] == 25
