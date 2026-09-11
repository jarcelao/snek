"""A NEAT-based Battlesnake policy and trainer."""

from .policy import encode_state, load_policy, network_action
from .training import (
    EvaluationOptions,
    EvaluationResult,
    TrainingOptions,
    TrainingResult,
    evaluate,
    train,
)

__all__ = [
    "encode_state",
    "evaluate",
    "EvaluationOptions",
    "EvaluationResult",
    "load_policy",
    "network_action",
    "train",
    "TrainingOptions",
    "TrainingResult",
]
