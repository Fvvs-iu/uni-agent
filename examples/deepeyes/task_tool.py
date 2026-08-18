"""Recipe-local multimodal crop tool for DeepEyes."""

from __future__ import annotations

import dataclasses
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from uni_agent.tools import Tool, ToolResult

from .image_utils import coerce_image, processor_safe_bbox


class ImageZoomInArgs(BaseModel):
    """Arguments exposed to the policy model."""

    bbox_2d: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Normalized 0-1000 coordinates [x1, y1, x2, y2].",
    )
    label: str | None = Field(default=None, description="Optional name of the inspected object or region.")

    model_config = ConfigDict(extra="forbid")


class ImageZoomInConfig(BaseModel):
    """Construction parameters that do not vary between calls."""

    min_dimension: int = Field(default=28, gt=0)
    max_aspect_ratio: float = Field(default=100.0, gt=1.0)
    coordinate_scale: float = Field(default=1000.0, gt=0.0)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0.0)

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass
class ImageZoomInResult(ToolResult):
    """Tool result with image payloads retained for the recipe Agent adapter."""

    images: list[Image.Image] = dataclasses.field(default_factory=list)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class ImageZoomInTool(Tool):
    """Crop a region from the source image while keeping Uni-Agent's Tool contract."""

    name = "image_zoom_in_tool"
    description = "Zoom in on a region of the source image. bbox_2d uses Qwen's normalized 0-1000 coordinates."
    args_model = ImageZoomInArgs
    config_model = ImageZoomInConfig

    def __init__(self, sandbox, *, image: Any, **kwargs: Any) -> None:
        super().__init__(sandbox, **kwargs)
        cfg: ImageZoomInConfig = self.config  # type: ignore[assignment]
        self._image = coerce_image(image, timeout=cfg.fetch_timeout_seconds)

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        del timeout
        cfg: ImageZoomInConfig = self.config  # type: ignore[assignment]
        try:
            bbox = processor_safe_bbox(
                args.get("bbox_2d"),
                image_size=self._image.size,
                min_dimension=cfg.min_dimension,
                max_aspect_ratio=cfg.max_aspect_ratio,
                coordinate_scale=cfg.coordinate_scale,
            )
        except ValueError as error:
            return ImageZoomInResult(
                text=f"Error: Could not zoom in: {error}",
                status="error",
                info={"success": False, "error": "invalid_bbox"},
            )

        crop = self._image.crop(bbox)
        label = args.get("label")
        label_text = f" with label {label}" if label else ""
        return ImageZoomInResult(
            text=f"Zoomed in on the image to the region {list(bbox)}{label_text}.",
            images=[crop],
            info={"success": True, "bbox": bbox, "crop_size": crop.size},
        )
