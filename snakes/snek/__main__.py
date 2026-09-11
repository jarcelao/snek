"""Train and evaluate the NEAT Battlesnake."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import EvaluationOptions, TrainingOptions, evaluate, load_policy, train


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and save a winning genome")
    train_parser.add_argument("--generations", type=_positive, default=20)
    train_parser.add_argument("--population", type=_positive, default=50)
    train_parser.add_argument("--games", type=_positive, default=5)
    train_parser.add_argument("--max-turns", type=_positive, default=250)
    train_parser.add_argument("--devin-depth", type=_positive, default=1)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--output-dir", type=Path, default=Path("train/snek"))
    train_parser.add_argument("--checkpoint-interval", type=_non_negative, default=5)
    train_parser.add_argument("--resume-checkpoint", type=Path)

    evaluate_parser = commands.add_parser(
        "evaluate", help="evaluate a saved winning genome"
    )
    evaluate_parser.add_argument("genome", type=Path)
    evaluate_parser.add_argument("--config", type=Path)
    evaluate_parser.add_argument("--games", type=_positive, default=20)
    evaluate_parser.add_argument("--max-turns", type=_positive, default=250)
    evaluate_parser.add_argument("--devin-depth", type=_positive, default=1)
    evaluate_parser.add_argument("--seed", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected command."""
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = train(
            TrainingOptions(
                generations=args.generations,
                population=args.population,
                games=args.games,
                max_turns=args.max_turns,
                devin_depth=args.devin_depth,
                seed=args.seed,
                output_dir=args.output_dir,
                checkpoint_interval=args.checkpoint_interval,
                resume_checkpoint=args.resume_checkpoint,
            )
        )
        print(f"Winner: {result.winner_path}")
        print(f"Fitness: {result.fitness:.3f}")
        return 0

    options = EvaluationOptions(
        games=args.games,
        max_turns=args.max_turns,
        devin_depth=args.devin_depth,
        seed=args.seed,
    )
    result = evaluate(
        load_policy(args.genome, args.config, max_turns=args.max_turns),
        options,
    )
    print(f"Wins: {result.wins}")
    print(f"Ties: {result.ties}")
    print(f"Losses: {result.losses}")
    print(f"Average fitness: {result.average_fitness:.3f}")
    print(f"Average turns: {result.average_turns:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
