"""DeepEyes visual question-answering task and reward."""

from __future__ import annotations

from .reward import DeepEyesJudgeConfig, DeepEyesRewardConfig, compute_score
from .task import DeepEyesTask, DeepEyesTaskConfig

__all__ = [
    "DeepEyesJudgeConfig",
    "DeepEyesRewardConfig",
    "DeepEyesTask",
    "DeepEyesTaskConfig",
    "compute_score",
]
