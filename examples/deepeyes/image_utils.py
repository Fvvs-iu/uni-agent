from __future__ import annotations

# Shared image decoding and safe-crop primitives for this recipe.

import base64
import binascii
import io
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image


def coerce_image(value: Any, *, timeout: float) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes | bytearray):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value:
            return coerce_image(value["bytes"], timeout=timeout)
        if "image" in value:
            return coerce_image(value["image"], timeout=timeout)
        if value.get("path"):
            return coerce_image(value["path"], timeout=timeout)
        image_url = value.get("image_url")
        if isinstance(image_url, dict) and "url" in image_url:
            return coerce_image(image_url["url"], timeout=timeout)
    if isinstance(value, str):
        if value.startswith("data:image"):
            try:
                _, encoded = value.split(",", 1)
                return coerce_image(base64.b64decode(encoded), timeout=timeout)
            except (ValueError, binascii.Error) as error:
                raise ValueError("invalid image data URL") from error
        if value.startswith(("http://", "https://")):
            import requests

            response = requests.get(value, timeout=timeout)
            response.raise_for_status()
            return coerce_image(response.content, timeout=timeout)
        parsed = urlparse(value)
        path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(value)
        if path.is_file():
            return Image.open(path).convert("RGB")
        raise ValueError(f"image path does not exist: {value}")
    raise TypeError(f"unsupported image payload: {type(value).__name__}")


def processor_safe_bbox(
    value: Any,
    *,
    image_size: tuple[int, int],
    min_dimension: int,
    max_aspect_ratio: float,
    coordinate_scale: float,
) -> tuple[int, int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("bbox_2d must contain four normalized coordinates")
    try:
        left, top, right, bottom = (float(coord) for coord in value)
    except (TypeError, ValueError) as error:
        raise ValueError("bbox_2d coordinates must be numeric") from error
    if not all(math.isfinite(coord) for coord in (left, top, right, bottom)):
        raise ValueError("bbox_2d coordinates must be finite")
    if left >= right or top >= bottom:
        raise ValueError("bbox_2d must satisfy x1 < x2 and y1 < y2")

    image_width, image_height = image_size
    left = left / coordinate_scale * image_width
    right = right / coordinate_scale * image_width
    top = top / coordinate_scale * image_height
    bottom = bottom / coordinate_scale * image_height
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(image_width), right)
    bottom = min(float(image_height), bottom)
    if left >= right or top >= bottom:
        raise ValueError(f"bbox_2d does not overlap image bounds {image_size}")

    width = right - left
    height = bottom - top
    if max(width, height) / min(width, height) > max_aspect_ratio:
        raise ValueError(f"bbox_2d aspect ratio exceeds {max_aspect_ratio:g}")

    target_width = min(float(image_width), max(width, float(min_dimension)))
    target_height = min(float(image_height), max(height, float(min_dimension)))
    if target_width < min_dimension or target_height < min_dimension:
        raise ValueError(f"image {image_size} cannot provide the minimum crop size {min_dimension}x{min_dimension}")

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    left = min(max(0.0, center_x - target_width / 2.0), image_width - target_width)
    top = min(max(0.0, center_y - target_height / 2.0), image_height - target_height)
    right = left + target_width
    bottom = top + target_height

    bbox = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(image_width, math.ceil(right)),
        min(image_height, math.ceil(bottom)),
    )
    if bbox[2] - bbox[0] < min_dimension or bbox[3] - bbox[1] < min_dimension:
        raise ValueError(f"processed crop is smaller than {min_dimension}x{min_dimension}")
    return bbox
