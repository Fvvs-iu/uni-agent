"""OpenAI message normalization helpers for DeepEyes."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image


def messages_for_gateway(messages: list[dict]) -> list[dict]:
    """Return serializable OpenAI messages with canonical image URL blocks."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("raw_prompt must be a non-empty message list")
    return [_message_for_gateway(message) for message in messages]


def _message_for_gateway(message: Any) -> dict:
    if not isinstance(message, dict):
        raise TypeError("raw_prompt messages must be mappings")
    normalized = dict(message)
    content = normalized.get("content", "")
    if isinstance(content, list):
        normalized["content"] = [_content_part_for_gateway(part) for part in content]
    elif not isinstance(content, str):
        raise TypeError("message content must be text or a content-part list")
    return normalized


def _content_part_for_gateway(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    image_keys = {"image", "image_url", "bytes"}
    if part.get("type") not in {"image", "image_url"} and not image_keys.intersection(part):
        return dict(part)
    if "image" in part:
        payload = part["image"]
    elif "bytes" in part:
        payload = {"bytes": part["bytes"]}
    else:
        image_url = part.get("image_url")
        payload = image_url.get("url") if isinstance(image_url, dict) else image_url
    return {"type": "image_url", "image_url": {"url": image_data_url(payload)}}


def image_data_url(value: Any) -> str:
    """Encode supported in-memory image values as an OpenAI data URL."""
    if isinstance(value, str):
        return value
    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        value = buffer.getvalue()
    elif isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, bytes | bytearray):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    raise TypeError(f"unsupported image payload: {type(value).__name__}")


def assistant_text(content: Any) -> str:
    """Flatten an OpenAI assistant content value to final-answer text."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", "")).strip() for part in content if isinstance(part, dict) and part.get("text")
    ).strip()
