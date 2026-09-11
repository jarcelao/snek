"""Run a local match with a snake adapter against Devious Devin."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
from types import ModuleType

from battlesnake_env import BattlesnakeEnv, OpponentPolicy, devious_devin
from snakes.sneklet import environment_policy

DEFAULT_SNAKE = "snakes.sneklet"


def _positive(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _adapter(module_name: str, parser: argparse.ArgumentParser) -> ModuleType:
    """Import and validate one snake adapter module."""
    if not module_name.startswith("snakes."):
        parser.error("--snake must name a module inside the snakes package")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        parser.error(f"cannot import snake adapter {module_name!r}: {error}")
    for name in ("configure_play_parser", "create_play_policy"):
        if not callable(getattr(module, name, None)):
            parser.error(f"snake adapter {module_name!r} must define callable {name}()")
    return module


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snake", default=DEFAULT_SNAKE)
    return parser


def _parser(adapter: ModuleType) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snake", default=DEFAULT_SNAKE, help="snake adapter module")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-turns", type=_positive, default=500)
    adapter.configure_play_parser(parser)
    return parser


def play(
    seed: int,
    *,
    policy: OpponentPolicy = environment_policy,
    snake_name: str = "Sneklet",
    render: bool = False,
    max_turns: int = 500,
) -> tuple[float, int, str]:
    """Play one seeded match and return its reward, turn, and result."""
    env = BattlesnakeEnv(
        devious_devin(max_depth=2),
        max_turns=max_turns,
        render_mode="ansi",
    )
    _, info = env.reset(seed=seed)
    while True:
        action = policy(info["state"], "snake-0")
        _, reward, terminated, truncated, info = env.step(action)
        if render:
            print(f"\nTurn {info['state'].turn}")
            print(env.render())
        if terminated or truncated:
            learner = info["state"].snakes[0]
            result = "won" if reward > 0 else "lost" if reward < 0 else "tied or reached the turn limit"
            print(
                f"Seed {seed}: {snake_name} {result} on turn "
                f"{info['state'].turn} ({learner.elimination or 'last snake alive'})."
            )
            return reward, info["state"].turn, result


def main(argv: Sequence[str] | None = None) -> int:
    """Run a match selected by the command-line snake adapter."""
    bootstrap = _bootstrap_parser()
    selected, _ = bootstrap.parse_known_args(argv)
    adapter = _adapter(selected.snake, bootstrap)
    parser = _parser(adapter)
    args = parser.parse_args(argv)
    policy = adapter.create_play_policy(args)
    if not callable(policy):
        parser.error(f"snake adapter {args.snake!r} did not create a callable policy")
    play(
        args.seed,
        policy=policy,
        snake_name=args.snake,
        render=args.render,
        max_turns=args.max_turns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
