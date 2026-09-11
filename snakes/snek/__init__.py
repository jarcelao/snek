"""A NEAT-based Battlesnake policy and trainer."""

import argparse
from pathlib import Path

from battlesnake_env import OpponentPolicy
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
    "configure_play_parser",
    "create_play_policy",
]


def configure_play_parser(parser: argparse.ArgumentParser) -> None:
    """Add NEAT model options to the local match parser."""
    parser.add_argument("--genome", type=Path, required=True, help="saved NEAT genome")
    parser.add_argument("--config", type=Path, help="optional NEAT configuration file")


def create_play_policy(args: argparse.Namespace) -> OpponentPolicy:
    """Load the selected NEAT policy for a local match."""
    return load_policy(args.genome, args.config, max_turns=args.max_turns)
