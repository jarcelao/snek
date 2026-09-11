"""Immutable Python implementation of the official standard rules pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

UP, DOWN, LEFT, RIGHT = range(4)
DELTAS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}
MAX_HEALTH = 100


class Elimination(StrEnum):
    NONE = ""
    COLLISION = "snake-collision"
    SELF_COLLISION = "snake-self-collision"
    OUT_OF_HEALTH = "out-of-health"
    HEAD_COLLISION = "head-collision"
    OUT_OF_BOUNDS = "wall-collision"
    HAZARD = "hazard"


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Snake:
    id: str
    body: tuple[Point, ...]
    health: int = MAX_HEALTH
    elimination: Elimination = Elimination.NONE
    eliminated_on_turn: int | None = None
    eliminated_by: str | None = None

    @property
    def alive(self) -> bool:
        return self.elimination is Elimination.NONE

    @property
    def head(self) -> Point:
        return self.body[0]


@dataclass(frozen=True)
class BoardState:
    width: int
    height: int
    turn: int
    food: tuple[Point, ...]
    hazards: tuple[Point, ...]
    snakes: tuple[Snake, ...]


@dataclass(frozen=True)
class Settings:
    food_spawn_chance: int = 15
    minimum_food: int = 1
    hazard_damage_per_turn: int = 14


def _shuffle(values: list[Point], rng: random.Random) -> list[Point]:
    result = values.copy()
    rng.shuffle(result)
    return result


def _unoccupied(state: BoardState, include_food: bool = True) -> list[Point]:
    occupied = {p for snake in state.snakes for p in snake.body}
    occupied.update(state.hazards)
    if include_food:
        occupied.update(state.food)
    return [
        Point(x, y)
        for x in range(state.width)
        for y in range(state.height)
        if Point(x, y) not in occupied
    ]


def initial_state(
    width: int, height: int, snake_ids: tuple[str, ...], rng: random.Random
) -> BoardState:
    """Make the official standard-map opening position."""
    mn, mid, mx = 1, (width - 1) // 2, width - 2
    corners = _shuffle(
        [Point(mn, mn), Point(mn, mx), Point(mx, mn), Point(mx, mx)], rng
    )
    cardinals = _shuffle(
        [Point(mn, mid), Point(mid, mn), Point(mid, mx), Point(mx, mid)], rng
    )
    starts = corners + cardinals if rng.randrange(2) == 0 else cardinals + corners
    if len(snake_ids) <= len(starts):
        snakes = tuple(Snake(id, (starts[i],) * 3) for i, id in enumerate(snake_ids))
    else:
        # The official standard map uses distributed placement for nine through
        # sixteen snakes on medium and larger boards. Keep every head on an even
        # tile and away from the centre, which preserves its placement invariant.
        candidates = [
            Point(x, y)
            for x in range(1, width - 1)
            for y in range(1, height - 1)
            if (x + y) % 2 == 0 and Point(x, y) != Point(mid, mid)
        ]
        candidates = _shuffle(candidates, rng)
        if len(candidates) < len(snake_ids):
            raise ValueError("standard map cannot place all snakes")
        snakes = tuple(
            Snake(id, (candidates[i],) * 3) for i, id in enumerate(snake_ids)
        )
    center = Point(mid, mid)
    food: list[Point] = []
    for snake in snakes:
        head = snake.head
        choices = []
        for point in (
            Point(head.x - 1, head.y - 1),
            Point(head.x - 1, head.y + 1),
            Point(head.x + 1, head.y - 1),
            Point(head.x + 1, head.y + 1),
        ):
            away = (
                (point.x < head.x < center.x)
                or (center.x < head.x < point.x)
                or (point.y < head.y < center.y)
                or (center.y < head.y < point.y)
            )
            corner = point.x in (0, width - 1) and point.y in (0, height - 1)
            if (
                0 <= point.x < width
                and 0 <= point.y < height
                and away
                and not corner
                and point != center
                and point not in food
            ):
                choices.append(point)
        if not choices:
            raise ValueError("standard map cannot place initial food")
        food.append(choices[rng.randrange(len(choices))])
    if center in {p for s in snakes for p in s.body} or center in food:
        raise ValueError("standard map cannot place center food")
    return BoardState(width, height, 0, tuple(food + [center]), (), snakes)


def _eliminate(snake: Snake, cause: Elimination, by: str | None, turn: int) -> Snake:
    return Snake(snake.id, snake.body, snake.health, cause, turn, by)


def _move(state: BoardState, moves: dict[str, int]) -> tuple[Snake, ...]:
    result = []
    for snake in state.snakes:
        if not snake.alive:
            result.append(snake)
            continue
        action = moves[snake.id]
        dx, dy = DELTAS[action]
        head = Point(snake.head.x + dx, snake.head.y + dy)
        result.append(Snake(snake.id, (head,) + snake.body[:-1], snake.health))
    return tuple(result)


def _feed(
    snakes: tuple[Snake, ...], food: tuple[Point, ...]
) -> tuple[tuple[Snake, ...], tuple[Point, ...]]:
    eaten = {snake.head for snake in snakes if snake.alive}
    remaining = tuple(point for point in food if point not in eaten)
    result = []
    for snake in snakes:
        if snake.alive and snake.head in eaten and snake.head in food:
            result.append(Snake(snake.id, snake.body + (snake.body[-1],), MAX_HEALTH))
        else:
            result.append(snake)
    return tuple(result), remaining


def _eliminations(state: BoardState, snakes: tuple[Snake, ...]) -> tuple[Snake, ...]:
    result = list(snakes)
    for index, snake in enumerate(result):
        if not snake.alive:
            continue
        if snake.health <= 0:
            result[index] = _eliminate(
                snake, Elimination.OUT_OF_HEALTH, None, state.turn + 1
            )
        elif any(
            p.x < 0 or p.x >= state.width or p.y < 0 or p.y >= state.height
            for p in snake.body
        ):
            result[index] = _eliminate(
                snake, Elimination.OUT_OF_BOUNDS, None, state.turn + 1
            )
    pending: dict[str, tuple[Elimination, str | None]] = {}
    alive = [snake for snake in result if snake.alive]
    for snake in alive:
        if snake.head in snake.body[1:]:
            pending[snake.id] = (Elimination.SELF_COLLISION, snake.id)
            continue
        bodies = sorted(alive, key=lambda item: len(item.body), reverse=True)
        other = next(
            (
                item
                for item in bodies
                if item.id != snake.id and snake.head in item.body[1:]
            ),
            None,
        )
        if other:
            pending[snake.id] = (Elimination.COLLISION, other.id)
            continue
        other = next(
            (
                item
                for item in bodies
                if item.id != snake.id
                and item.head == snake.head
                and len(snake.body) <= len(item.body)
            ),
            None,
        )
        if other:
            pending[snake.id] = (Elimination.HEAD_COLLISION, other.id)
    return tuple(
        _eliminate(snake, *pending[snake.id], state.turn + 1)
        if snake.id in pending
        else snake
        for snake in result
    )


def _spawn_food(
    state: BoardState, settings: Settings, rng: random.Random
) -> tuple[Point, ...]:
    needed = max(0, settings.minimum_food - len(state.food))
    if (
        needed == 0
        and settings.food_spawn_chance > 0
        and 100 - rng.randrange(100) < settings.food_spawn_chance
    ):
        needed = 1
    available = _shuffle(_unoccupied(state, include_food=True), rng)
    return state.food + tuple(available[:needed])


def step(
    state: BoardState, moves: dict[str, int], settings: Settings, rng: random.Random
) -> tuple[BoardState, bool]:
    """Apply the standard rules pipeline and standard-map food update."""
    if sum(snake.alive for snake in state.snakes) <= 1:
        return state, True
    alive = [snake for snake in state.snakes if snake.alive]
    if set(moves) != {snake.id for snake in alive} or any(
        action not in DELTAS for action in moves.values()
    ):
        raise ValueError("provide one valid action for every living snake")
    moved = _move(state, moves)
    starved = tuple(
        Snake(
            s.id,
            s.body,
            s.health - 1,
            s.elimination,
            s.eliminated_on_turn,
            s.eliminated_by,
        )
        if s.alive
        else s
        for s in moved
    )
    damaged = []
    for snake in starved:
        if snake.alive and snake.head in state.hazards and snake.head not in state.food:
            health = max(0, snake.health - settings.hazard_damage_per_turn)
            damaged.append(Snake(snake.id, snake.body, health))
        else:
            damaged.append(snake)
    fed, food = _feed(tuple(damaged), state.food)
    eliminated = _eliminations(state, fed)
    advanced = BoardState(
        state.width, state.height, state.turn + 1, food, state.hazards, eliminated
    )
    return BoardState(
        advanced.width,
        advanced.height,
        advanced.turn,
        _spawn_food(advanced, settings, rng),
        advanced.hazards,
        advanced.snakes,
    ), False
