"""Recipe-local image ReAct agent used by DeepEyes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field

from uni_agent.agents import Agent, AgentConfig, AgentResult
from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.agents.registry import register_agent
from uni_agent.tools import ToolResult, Toolbox

from .image_utils import coerce_image
from .message_utils import assistant_text, image_data_url, messages_for_gateway
from .task_tool import ImageZoomInConfig, ImageZoomInResult, ImageZoomInTool

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You can call functions to assist with the user query. "
    "Important: You must call only one function at a time."
)


class ImageZoomReActConfig(AgentConfig):
    """DeepEyes policy loop and image-tool settings."""

    name: str = "image_zoom_react"
    max_turns: int = Field(default=4, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    action_timeout_seconds: float | None = Field(default=None, gt=0.0)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    image_tool: ImageZoomInConfig = Field(default_factory=ImageZoomInConfig)


@register_agent("image_zoom_react")
class ImageZoomReActAgent(Agent):
    """Sequential multimodal ReAct loop whose only action is image zoom."""

    config_model = ImageZoomReActConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: ImageZoomReActConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("image_zoom_react: config.model.base_url is not set")

        prompt = _with_system_prompt(messages, cfg.system_prompt)
        source_image = _first_image(prompt, timeout=cfg.image_tool.fetch_timeout_seconds)
        transcript = messages_for_gateway(prompt)
        logger.info(
            "DeepEyes agent start: source_image=%sx%s messages=%s max_turns=%s",
            source_image.width,
            source_image.height,
            len(transcript),
            cfg.max_turns,
        )
        tool = ImageZoomInTool(
            sandbox,
            image=source_image,
            **cfg.image_tool.model_dump(),
        )
        toolbox = Toolbox([tool])
        request_params = cfg.model.sampling_params()
        request_params["tool_choice"] = "auto"
        model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=request_params,
            tools_schemas=toolbox.schemas(),
            timeout=cfg.request_timeout_seconds,
        )

        info: dict[str, Any] = {
            "steps": 0,
            "tool_calls": 0,
            "tool_successes": 0,
            "tool_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        final_answer: str | None = None
        termination_reason = "unknown"
        try:
            async with toolbox.entered(retry=1, timeout=60):
                for turn_index in range(cfg.max_turns):
                    info["steps"] = turn_index + 1
                    logger.info("DeepEyes agent turn %s/%s", turn_index + 1, cfg.max_turns)
                    content, tool_calls, generation = await model.query(transcript)
                    info["prompt_tokens"] += int(generation.get("prompt_tokens", 0))
                    info["completion_tokens"] += int(generation.get("completion_tokens", 0))
                    info["total_tokens"] = info["prompt_tokens"] + info["completion_tokens"]

                    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    transcript.append(assistant_message)
                    logger.info("DeepEyes assistant:\n%s", content)

                    finish_reason = generation.get("finish_reason")
                    if not tool_calls:
                        final_answer = assistant_text(content)
                        termination_reason = "token_limit" if finish_reason == "length" else "finished"
                        break

                    info["tool_calls"] += len(tool_calls)
                    if turn_index + 1 >= cfg.max_turns:
                        info["tool_errors"] += len(tool_calls)
                        termination_reason = "max_turns"
                        logger.warning("DeepEyes exhausted max_turns=%s with pending tool calls", cfg.max_turns)
                        break

                    for tool_call in tool_calls:
                        function = tool_call.get("function") or {}
                        tool_name = str(function.get("name", ""))
                        logger.info("DeepEyes tool call %s: %s", tool_name, function.get("arguments"))
                        result = await toolbox.call(
                            tool_name,
                            function.get("arguments"),
                            timeout=cfg.action_timeout_seconds,
                        )
                        successful = (
                            result.status == "ok"
                            and isinstance(result, ImageZoomInResult)
                            and result.info.get("success") is True
                        )
                        if successful:
                            info["tool_successes"] += 1
                        else:
                            info["tool_errors"] += 1
                        transcript.append(_tool_message(str(tool_call.get("id", "")), result))
                        logger.info("DeepEyes tool result status=%s: %s", result.status, result.text or "")
                else:
                    termination_reason = "max_turns"
        finally:
            await model.aclose()

        finished = termination_reason == "finished"
        info["termination_reason"] = termination_reason
        logger.info(
            "DeepEyes agent done: reason=%s turns=%s calls=%s successes=%s errors=%s final_answer=%s",
            termination_reason,
            info["steps"],
            info["tool_calls"],
            info["tool_successes"],
            info["tool_errors"],
            final_answer,
        )
        return AgentResult(
            output={"final_answer": final_answer, "termination_reason": termination_reason},
            transcript=transcript,
            info=info,
            finished=finished,
        )


def _with_system_prompt(messages: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("DeepEyes task prompt must be a non-empty message list")
    copied = [dict(message) for message in messages]
    if copied[0].get("role") != "system":
        copied.insert(0, {"role": "system", "content": system_prompt})
    return copied


def _first_image(messages: list[dict[str, Any]], *, timeout: float):
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if "image" in part:
                return coerce_image(part["image"], timeout=timeout)
            if "bytes" in part:
                return coerce_image({"bytes": part["bytes"]}, timeout=timeout)
            if part.get("type") == "image_url" or "image_url" in part:
                return coerce_image({"image_url": part.get("image_url")}, timeout=timeout)
    raise ValueError("DeepEyes prompt does not contain a source image")


def _tool_message(tool_call_id: str, result: ToolResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.text is not None:
        content.append({"type": "text", "text": str(result.text)})
    if isinstance(result, ImageZoomInResult):
        content.extend(
            {"type": "image_url", "image_url": {"url": image_data_url(image)}}
            for image in result.images
        )
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
