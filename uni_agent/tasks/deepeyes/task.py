"""DeepEyes visual question-answering task."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import DeepEyesRewardConfig, compute_score

logger = logging.getLogger(__name__)


class DeepEyesTaskConfig(TaskConfig):
    """One DeepEyes sample plus the configured image-solving Agent."""

    name: str = "deepeyes"
    question: str = Field(min_length=1)
    ground_truth: str
    data_source: str = "deepeyes"
    reward: DeepEyesRewardConfig = Field(default_factory=DeepEyesRewardConfig)


@register_task("deepeyes")
class DeepEyesTask(Task):
    """Run one image question and score the final answer with the DeepEyes Judge."""

    config_model = DeepEyesTaskConfig

    async def run(self) -> TaskResult:
        cfg: DeepEyesTaskConfig = self.config  # type: ignore[assignment]
        logger.info("DeepEyes task start: question=%s", cfg.question.strip())

        async with self.build_sandbox() as sandbox:
            agent_result = await self.build_agent().run(
                sandbox=sandbox,
                messages=cfg.prompt,
                workdir=None,
            )

        final_answer_value = agent_result.output.get("final_answer")
        final_answer = final_answer_value if isinstance(final_answer_value, str) else None
        reward_context: dict[str, Any] = {
            "question": cfg.question,
            "finished": agent_result.finished is True,
            "final_answer": final_answer,
            "tool_calls": agent_result.info.get("tool_calls", 0),
            "tool_successes": agent_result.info.get("tool_successes", 0),
            "tool_errors": agent_result.info.get("tool_errors", 0),
        }
        score = await asyncio.to_thread(
            compute_score,
            cfg.data_source,
            final_answer or "",
            cfg.ground_truth,
            reward_context,
            reward_config=cfg.reward,
        )

        extra_info: dict[str, Any] = {
            **score,
            "termination_reason": agent_result.info.get("termination_reason", "unknown"),
            "steps": agent_result.info.get("steps", 0),
            "prompt_tokens": agent_result.info.get("prompt_tokens", 0),
            "completion_tokens": agent_result.info.get("completion_tokens", 0),
            "total_tokens": agent_result.info.get("total_tokens", 0),
        }
        logger.info(
            "DeepEyes task done: reward=%s acc=%s finished=%s calls=%s successes=%s errors=%s reason=%s",
            score["score"],
            score["acc"],
            agent_result.finished,
            score["tool_calls"],
            score["tool_successes"],
            score["tool_errors"],
            extra_info["termination_reason"],
        )
        return TaskResult(
            reward=float(score["score"]),
            accuracy=float(score["acc"]),
            finished=agent_result.finished is True,
            extra_info=extra_info,
        )
