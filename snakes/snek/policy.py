"""Feature encoding and policy loading for the NEAT snake."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import neat
import numpy as np

from battlesnake_env import BoardState, OpponentPolicy
from battlesnake_env.rules import DELTAS, MAX_HEALTH


class Network(Protocol):
    """A network that supplies one score for each move."""

    def activate(self, inputs: Sequence[float]) -> Sequence[float]: ...


def _relative(source, targets, scale: float) -> tuple[float, float]:
    if not targets:
        return 0.0, 0.0
    target = min(
        targets,
        key=lambda point: (
            abs(point.x - source.x) + abs(point.y - source.y),
            point.x,
            point.y,
        ),
    )
    return (target.x - source.x) / scale, (target.y - source.y) / scale


def encode_state(state: BoardState, snake_id: str, *, max_turns: int) -> np.ndarray:
    """Encode one snake's view as 20 normalized values."""
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    try:
        snake = next(item for item in state.snakes if item.id == snake_id)
    except StopIteration as error:
        raise ValueError(f"unknown snake id: {snake_id}") from error
    if not snake.alive:
        raise ValueError(f"snake is not alive: {snake_id}")

    head = snake.head
    scale = float(max(state.width - 1, state.height - 1, 1))
    occupied = set(snake.body[1:])
    occupied.update(
        point
        for item in state.snakes
        if item.alive and item.id != snake_id
        for point in item.body
    )
    safe: list[float] = []
    walls: list[float] = []
    obstacle_rays: list[float] = []
    for action in range(4):
        dx, dy = DELTAS[action]
        adjacent = type(head)(head.x + dx, head.y + dy)
        safe.append(
            float(
                0 <= adjacent.x < state.width
                and 0 <= adjacent.y < state.height
                and adjacent not in snake.body[1:]
            )
        )

        steps_to_wall = 0
        point = head
        while 0 <= point.x + dx < state.width and 0 <= point.y + dy < state.height:
            steps_to_wall += 1
            point = type(head)(point.x + dx, point.y + dy)
        walls.append(steps_to_wall / scale)

        obstacle_distance = steps_to_wall + 1
        point = head
        for distance in range(1, steps_to_wall + 1):
            point = type(head)(point.x + dx, point.y + dy)
            if point in occupied or point in state.hazards:
                obstacle_distance = distance
                break
        obstacle_rays.append(obstacle_distance / (scale + 1.0))

    opponents = [item for item in state.snakes if item.alive and item.id != snake_id]
    nearest_opponent = min(
        opponents,
        key=lambda item: (
            abs(item.head.x - head.x) + abs(item.head.y - head.y),
            item.id,
        ),
        default=None,
    )
    food_dx, food_dy = _relative(head, state.food, scale)
    opponent_dx, opponent_dy = _relative(
        head,
        () if nearest_opponent is None else (nearest_opponent.head,),
        scale,
    )
    relative_length = 0.0
    if nearest_opponent is not None:
        relative_length = (len(snake.body) - len(nearest_opponent.body)) / float(
            state.width * state.height
        )

    values = (
        *safe,
        *walls,
        *obstacle_rays,
        food_dx,
        food_dy,
        opponent_dx,
        opponent_dy,
        snake.health / MAX_HEALTH,
        len(snake.body) / float(state.width * state.height),
        relative_length,
        min(1.0, state.turn / max_turns),
    )
    return np.asarray(values, dtype=np.float32)


def network_action(
    network: Network,
    state: BoardState,
    snake_id: str,
    *,
    max_turns: int,
) -> int:
    """Select the highest-scoring advisory safe action."""
    features = encode_state(state, snake_id, max_turns=max_turns)
    outputs = tuple(network.activate(tuple(float(feature) for feature in features)))
    if len(outputs) != 4:
        raise ValueError("network must return four outputs")
    candidates = [action for action in range(4) if features[action] == 1.0]
    if not candidates:
        candidates = list(range(4))
    return max(candidates, key=lambda action: (outputs[action], -action))


def _load_config(path: Path) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def load_policy(
    genome_path: str | Path,
    config_path: str | Path | None = None,
    *,
    max_turns: int = 250,
) -> OpponentPolicy:
    """Load a saved winning genome as an environment policy."""
    genome_file = Path(genome_path)
    effective_config = (
        Path(config_path)
        if config_path is not None
        else genome_file.with_name("config.ini")
    )
    with genome_file.open("rb") as stream:
        genome = pickle.load(stream)
    network = neat.nn.FeedForwardNetwork.create(genome, _load_config(effective_config))

    def policy(state: BoardState, snake_id: str) -> int:
        return network_action(network, state, snake_id, max_turns=max_turns)

    return policy
