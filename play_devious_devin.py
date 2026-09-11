"""Run a local match: the template snake versus Devious Devin."""

from __future__ import annotations

import argparse

from battlesnake_env import BattlesnakeEnv, devious_devin
from snek import environment_policy


def play(seed: int, *, render: bool = False) -> tuple[float, int, str]:
    """Play one seeded match and return its reward, turn, and result."""
    env = BattlesnakeEnv(devious_devin(max_depth=2), render_mode="ansi")
    _, info = env.reset(seed=seed)
    while True:
        action = environment_policy(info["state"], "snake-0")
        _, reward, terminated, truncated, info = env.step(action)
        if render:
            print(f"\nTurn {info['state'].turn}")
            print(env.render())
        if terminated or truncated:
            learner = info["state"].snakes[0]
            result = "won" if reward > 0 else "lost" if reward < 0 else "tied or reached the turn limit"
            print(
                f"Seed {seed}: template snake {result} on turn "
                f"{info['state'].turn} ({learner.elimination or 'last snake alive'})."
            )
            return reward, info["state"].turn, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    play(args.seed, render=args.render)
