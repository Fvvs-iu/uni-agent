"""LLM-as-a-Judge reward used by DeepEyes."""

from __future__ import annotations

import logging
import os
import random
import re
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_BASE = "http://127.0.0.1:18901/v1"
ACCURACY_WEIGHT = 0.8
FORMAT_WEIGHT = 0.2
TOOL_WEIGHT = 1.2
MAX_ANSWER_CHARS = 1000


@lru_cache(maxsize=1)
def _get_judge_client() -> tuple[Any | None, str]:
    base_url = os.environ.get("LLM_AS_A_JUDGE_BASE", DEFAULT_JUDGE_BASE).rstrip("/")
    model_name = os.environ.get("LLM_AS_A_JUDGE_MODEL", "")
    timeout = float(os.environ.get("LLM_AS_A_JUDGE_TIMEOUT_SECONDS", "120"))
    max_retries = int(os.environ.get("LLM_AS_A_JUDGE_MAX_RETRIES", "2"))
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("LLM_AS_A_JUDGE_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            # Reward workers inherit the host's proxy environment.  The Judge
            # is a local service and must never be routed through that proxy.
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )
        if not model_name:
            models = client.models.list()
            if models.data:
                model_name = models.data[0].id
        if not model_name:
            raise RuntimeError(f"judge service at {base_url} returned no models")
        return client, model_name
    except Exception as error:  # noqa: BLE001 - handled by strict/optional policy below
        if _strict_judge():
            raise RuntimeError(f"DeepEyes judge initialization failed for {base_url}: {error}") from error
        logger.warning("DeepEyes judge unavailable; returning zero rewards: %s", error)
        return None, ""


def check_judge() -> str:
    """Run a real semantic judgement and return the selected model name."""
    client, model_name = _get_judge_client()
    if client is None or not model_name:
        raise RuntimeError("DeepEyes judge is unavailable")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _judge_system_prompt()},
                {
                    "role": "user",
                    "content": _judge_user_prompt(
                        question="What is two plus two?",
                        ground_truth="4",
                        answer="4",
                    ),
                },
            ],
            seed=0,
            temperature=0.1,
            max_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as error:  # noqa: BLE001 - preflight must fail closed
        raise RuntimeError(f"DeepEyes judge completion preflight failed: {error}") from error
    judgement = (response.choices[0].message.content or "").strip()
    if not re.search(r"\bCORRECT\b", judgement, re.IGNORECASE):
        raise RuntimeError(f"DeepEyes judge completion preflight returned {judgement!r}, expected CORRECT")
    return model_name


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict[str, float]:
    """Return DeepEyes accuracy, format, and correct-tool-use reward components."""
    del data_source
    reward_context = extra_info or {}
    question_text = reward_context.get("question", "")
    if not question_text:
        raise ValueError("DeepEyes reward requires extra_info.question")

    finished = reward_context.get("finished") is not False
    tool_call_count = _nonnegative_int(reward_context.get("tool_calls"))
    tool_success_count = _nonnegative_int(reward_context.get("tool_successes"))
    tool_error_count = _nonnegative_int(reward_context.get("tool_errors"))
    if tool_call_count is None:
        tool_call_count = int(_has_tool_usage(solution_str))
    if tool_success_count is None:
        tool_success_count = 0
    if tool_error_count is None:
        tool_error_count = 0

    structured_answer = reward_context.get("final_answer")
    has_structured_answer = isinstance(structured_answer, str)
    answer_source = structured_answer if has_structured_answer else solution_str
    answer_text, format_error, answer_tags = _extract_answer_details(
        answer_source,
        allow_prefilled_think=has_structured_answer,
    )
    if not finished:
        return _reward_result(
            accuracy_reward=0.0,
            format_error=True,
            has_successful_tool_usage=_has_successful_tool_usage(tool_success_count),
            finished=False,
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count,
            tool_error_count=tool_error_count,
            answer_tags=answer_tags,
        )

    # Keep this guard before the Judge request.  Besides being invalid output
    # according to the original DeepEyes reward, a very long answer can push
    # the Judge prompt beyond its serving context window and fail the entire
    # rollout instead of simply receiving the intended format penalty.
    if len(answer_text) >= MAX_ANSWER_CHARS:
        return _reward_result(
            accuracy_reward=0.0,
            format_error=True,
            has_successful_tool_usage=_has_successful_tool_usage(tool_success_count),
            finished=True,
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count,
            tool_error_count=tool_error_count,
            answer_tags=answer_tags,
        )

    client, model_name = _get_judge_client()
    if client is None or not model_name:
        return _reward_result(
            accuracy_reward=0.0,
            format_error=format_error,
            has_successful_tool_usage=_has_successful_tool_usage(tool_success_count),
            finished=True,
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count,
            tool_error_count=tool_error_count,
            answer_tags=answer_tags,
            score_override=0.0,
        )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _judge_system_prompt()},
                {
                    "role": "user",
                    "content": _judge_user_prompt(
                        question=question_text,
                        ground_truth=str(ground_truth),
                        answer=answer_text,
                    ),
                },
            ],
            seed=random.randint(0, 1_000_000),
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as error:  # noqa: BLE001 - policy is controlled by DEEPEYES_JUDGE_STRICT
        if _strict_judge():
            raise RuntimeError(f"DeepEyes judge request failed: {error}") from error
        logger.warning("DeepEyes judge request failed; returning zero rewards: %s", error)
        return _reward_result(
            accuracy_reward=0.0,
            format_error=format_error,
            has_successful_tool_usage=_has_successful_tool_usage(tool_success_count),
            finished=True,
            tool_call_count=tool_call_count,
            tool_success_count=tool_success_count,
            tool_error_count=tool_error_count,
            answer_tags=answer_tags,
            score_override=0.0,
        )
    judgement = (response.choices[0].message.content or "").strip()
    if re.search(r"\bINCORRECT\b", judgement, re.IGNORECASE):
        accuracy_reward = 0.0
    elif re.search(r"\bCORRECT\b", judgement, re.IGNORECASE):
        accuracy_reward = 1.0
    else:
        raise ValueError(f"Judge returned neither CORRECT nor INCORRECT: {judgement!r}")

    return _reward_result(
        accuracy_reward=accuracy_reward,
        format_error=format_error,
        has_successful_tool_usage=_has_successful_tool_usage(tool_success_count),
        finished=True,
        tool_call_count=tool_call_count,
        tool_success_count=tool_success_count,
        tool_error_count=tool_error_count,
        answer_tags=answer_tags,
    )


def _reward_result(
    *,
    accuracy_reward: float,
    format_error: bool,
    has_successful_tool_usage: bool,
    finished: bool,
    tool_call_count: int,
    tool_success_count: int,
    tool_error_count: int,
    answer_tags: bool,
    score_override: float | None = None,
) -> dict[str, float]:
    # Match the original DeepEyes intent: grant the conditional tool bonus only
    # after active perception actually succeeds. A serialized tool marker or a
    # failed execution alone must not receive the bonus.
    tool_reward = 1.0 if has_successful_tool_usage and accuracy_reward > 0.5 else 0.0
    format_reward = -1.0 if format_error else 0.0
    final_score = ACCURACY_WEIGHT * accuracy_reward + FORMAT_WEIGHT * format_reward + TOOL_WEIGHT * tool_reward
    return {
        "score": final_score if score_override is None else score_override,
        "acc": accuracy_reward,
        "format": format_reward,
        "tool": tool_reward,
        "finished": float(finished),
        "tool_calls": float(tool_call_count),
        "tool_successes": float(tool_success_count),
        "tool_errors": float(tool_error_count),
        "answer_tags": float(answer_tags),
    }


def _nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, result)


def _has_successful_tool_usage(tool_successes: int) -> bool:
    return tool_successes > 0


def _has_tool_usage(solution_str: str) -> bool:
    return bool(
        re.search(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
        or re.search(r"<tool_response>.*?</tool_response>", solution_str, re.DOTALL)
    )


def _strict_judge() -> bool:
    value = os.environ.get("DEEPEYES_JUDGE_STRICT", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("DEEPEYES_JUDGE_STRICT must be a boolean value")


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    answer, format_error, _ = _extract_answer_details(solution_str, allow_prefilled_think=False)
    return answer, format_error


def _extract_answer_details(
    solution_str: str,
    *,
    allow_prefilled_think: bool,
) -> tuple[str, bool, bool]:
    final_content = _last_assistant_content(solution_str)
    think_open_count = final_content.count("<think>")
    think_close_count = final_content.count("</think>")
    # Qwen3.5's native chat template pre-fills ``<think>`` in the generation
    # prompt. It is therefore part of prompt_ids, while the structured final
    # assistant content starts after that token and contains only ``</think>``.
    # Treat exactly one such missing opener as a valid native turn boundary.
    prefilled_think = allow_prefilled_think and think_open_count == 0 and think_close_count == 1
    format_error = think_open_count != think_close_count and not prefilled_think
    answer_region = (
        final_content.split("</think>")[-1].strip() if "</think>" in final_content else final_content.strip()
    )
    answer_region = answer_region.replace("<|im_end|>", "").strip()
    answer_open_count = answer_region.count("<answer>")
    answer_close_count = answer_region.count("</answer>")
    match = re.search(r"<answer>(.*?)</answer>", answer_region, re.DOTALL)
    answer_tags = answer_open_count == 1 and answer_close_count == 1 and match is not None
    if answer_open_count != answer_close_count:
        format_error = True

    if match:
        answer = match.group(1).strip()
    else:
        # Keep a usable plain-text answer for semantic judging, but preserve
        # the original DeepEyes format penalty when <answer> tags are absent.
        format_error = True
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$",
            answer_region,
            re.DOTALL | re.MULTILINE,
        )
        if tool_response_match:
            answer = tool_response_match.group(1).strip()
        else:
            answer = re.sub(r"<tool_call>.*?</tool_call>", "", answer_region, flags=re.DOTALL)
            answer = re.sub(r"<tool_response>.*?</tool_response>", "", answer, flags=re.DOTALL)
            answer = re.sub(r"\b(user|assistant)\b", "", answer).strip()
    if not answer:
        format_error = True
        answer = final_content.strip()
    return answer, format_error, answer_tags


def _last_assistant_content(solution_str: str) -> str:
    matches = re.findall(
        r"<\|im_start\|>assistant\s*\n?(.*?)(?=<\|im_end\|>)",
        solution_str,
        re.DOTALL,
    )
    return matches[-1].strip() if matches else solution_str.strip()


def _judge_system_prompt() -> str:
    return (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        "standard answer.\n"
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )


def _judge_user_prompt(*, question: str, ground_truth: str, answer: str) -> str:
    return (
        "I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
        "answer is correct.\n\n"
        "---\n"
        "**Example 1:**\n"
        "[Question]: Is the countertop tan or blue?\n"
        "[Standard Answer]: The countertop is tan.\n"
        "[Model's Answer]: tan\n"
        "[Your Judgement]: CORRECT\n"
        "---\n"
        "**Example 2:**\n"
        "[Question]: Is the man phone both blue and closed?\n"
        "[Standard Answer]: Yes, the man phone is both blue and closed.\n"
        "[Model's Answer]: No.\n"
        "[Your Judgement]: INCORRECT\n"
        "---\n"
        "**Task:**\n"
        f"[Question]: {question}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {answer}\n"
        "[Your Judgement]:"
    )


if __name__ == "__main__":
    selected_model = check_judge()
    print(f"DeepEyes judge ready: {selected_model}")
