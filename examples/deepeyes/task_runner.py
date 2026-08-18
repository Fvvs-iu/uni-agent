"""Strict async-recipe wrapper around Uni-Agent's generic task runner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from uni_agent.framework.task_runner import run_task
from uni_agent.tasks import TaskResult

# Register the recipe-local implementations inside every ray_task worker before
# the generic runner resolves the serialized TaskConfig.
from . import task_agent as _task_agent  # noqa: F401
from . import task as _task  # noqa: F401

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)

_METRIC_KEYS = (
    "format",
    "tool",
    "tool_calls",
    "tool_successes",
    "tool_errors",
    "answer_tags",
    "steps",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


async def run_deepeyes_task(
    *,
    session: SessionHandle,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    tools_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> TaskResult:
    """Resolve a DeepEyesTask, execute it, then strictly publish its reward."""
    kwargs.pop("report_reward", None)
    result = await run_task(
        session=session,
        raw_prompt=raw_prompt,
        sample_index=sample_index,
        tools_kwargs=tools_kwargs,
        report_reward=False,
        **kwargs,
    )
    await _post_reward_info_strict(session.reward_info_url, result)
    return result


async def _post_reward_info_strict(reward_info_url: str | None, result: TaskResult) -> None:
    if not reward_info_url:
        raise ValueError("DeepEyes task runner requires session.reward_info_url")
    if result.finished is not None and type(result.finished) is not bool:
        raise ValueError("TaskResult.finished must be a bool or None")

    reward_info: dict[str, Any] = {"reward": float(result.reward)}
    if result.accuracy is not None:
        reward_info["acc"] = float(result.accuracy)
    if result.finished is not None:
        reward_info["finished"] = result.finished
    for key in _METRIC_KEYS:
        value = (result.extra_info or {}).get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            reward_info[key] = float(value)

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
        async with client.post(reward_info_url, json={"reward_info": reward_info}) as response:
            response.raise_for_status()
    logger.info("DeepEyes reward posted: %s", reward_info)
