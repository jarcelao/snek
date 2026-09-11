"""
The Devious Devin paranoid-minimax opponent.

Adapted from Corey Alexander's Battlesnake of the same name:
https://github.com/coreyja/battlesnake-rs/
"""

from __future__ import annotations

from collections import deque
import random

from .env import OpponentPolicy
from .rules import DELTAS, DOWN, LEFT, RIGHT, UP, BoardState, Point, Settings, Snake, step

_ACTION_ORDER = (UP, DOWN, LEFT, RIGHT)
_FROZEN_FOOD_SETTINGS = Settings(food_spawn_chance=0, minimum_food=0)
_NO_PATH = -1_000_000


class _SearchLimitReached(Exception):
    """Stop a search that used its allowed node count."""


def _living(state: BoardState) -> tuple[Snake, ...]:
    return tuple(snake for snake in state.snakes if snake.alive)


def _legal_actions(state: BoardState, snake: Snake) -> tuple[int, ...]:
    """Return Devin's in-board, non-reversing moves in stable order."""
    neck = snake.body[1] if len(snake.body) > 1 else None
    actions = []
    for action in _ACTION_ORDER:
        dx, dy = DELTAS[action]
        point = Point(snake.head.x + dx, snake.head.y + dy)
        if 0 <= point.x < state.width and 0 <= point.y < state.height and point != neck:
            actions.append(action)
    return tuple(actions) or (UP,)


def _distance(state: BoardState, start: Point, targets: tuple[Point, ...]) -> int | None:
    """Find the shortest path while treating snake bodies as blocked."""
    if not targets:
        return None
    target_set = set(targets)
    blocked = {point for snake in state.snakes for point in snake.body}
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        point, distance = queue.popleft()
        if point in target_set:
            return distance
        for dx, dy in DELTAS.values():
            neighbor = Point(point.x + dx, point.y + dy)
            if (neighbor in visited or neighbor in blocked and neighbor not in target_set
                    or not 0 <= neighbor.x < state.width or not 0 <= neighbor.y < state.height):
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _score(state: BoardState, snake_id: str, completed_turns: int) -> tuple[int, int, int, int]:
    """Return Devin's ordered terminal or positional score."""
    alive = _living(state)
    me = next(snake for snake in state.snakes if snake.id == snake_id)
    if len(alive) <= 1:
        if me.alive:
            return (4, -completed_turns, 0, 0)
        if not alive:
            return (1, 0, completed_turns, 0)
        return (0, -len(alive), completed_turns, 0)

    opponents = tuple(snake for snake in alive if snake.id != snake_id)
    if not me.alive:
        return (0, -len(alive), completed_turns, 0)
    max_opponent_length = max(len(snake.body) for snake in opponents)
    length_difference = len(me.body) - max_opponent_length
    health = max(me.health, 50)
    if max_opponent_length >= len(me.body) or me.health < 20:
        distance = _distance(state, me.head, state.food)
        return (2, length_difference, _NO_PATH if distance is None else -distance, health)

    distance = _distance(state, me.head, tuple(snake.head for snake in opponents))
    return (3, _NO_PATH if distance is None else -distance, max(length_difference, 4), health)


def _advance(state: BoardState, moves: dict[str, int]) -> BoardState:
    """Advance one simultaneous turn without adding new food."""
    return step(state, moves, _FROZEN_FOOD_SETTINGS, random.Random(0))[0]


def _one_ply_fallback(state: BoardState, snake_id: str) -> int:
    """Choose the best immediate result against stable opponent moves."""
    me = next(snake for snake in state.snakes if snake.id == snake_id)
    opponent_moves = {
        snake.id: _legal_actions(state, snake)[0]
        for snake in _living(state)
        if snake.id != snake_id
    }
    best_action = _legal_actions(state, me)[0]
    best_score: tuple[int, int, int, int] | None = None
    for action in _legal_actions(state, me):
        next_state = _advance(state, opponent_moves | {snake_id: action})
        score = _score(next_state, snake_id, 1)
        if best_score is None or score > best_score:
            best_action, best_score = action, score
    return best_action


def devious_devin(*, max_depth: int = 2, max_nodes: int = 50_000) -> OpponentPolicy:
    """Create a deterministic Devious Devin policy.

    ``max_depth`` is the number of complete simultaneous turns to search.
    ``max_nodes`` limits the work for one policy call.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    if not isinstance(max_nodes, int) or isinstance(max_nodes, bool) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")

    def policy(state: BoardState, snake_id: str) -> int:
        snakes_by_id = {snake.id: snake for snake in state.snakes}
        if snake_id not in snakes_by_id:
            raise ValueError("snake_id must identify a snake on the board")
        if not snakes_by_id[snake_id].alive:
            return UP

        completed: tuple[int, tuple[int, int, int, int]] | None = None
        nodes = 0
        for depth_limit in range(1, max_depth + 1):
            def search_round(node: BoardState, turns: int) -> tuple[int, int, int, int]:
                nonlocal nodes
                nodes += 1
                if nodes > max_nodes:
                    raise _SearchLimitReached
                if turns == depth_limit or len(_living(node)) <= 1:
                    return _score(node, snake_id, turns)

                players = tuple(snake for snake in _living(node) if snake.id == snake_id)
                players += tuple(snake for snake in _living(node) if snake.id != snake_id)

                def choose(index: int, moves: dict[str, int]) -> tuple[int, int, int, int]:
                    nonlocal nodes
                    nodes += 1
                    if nodes > max_nodes:
                        raise _SearchLimitReached
                    if index == len(players):
                        return search_round(_advance(node, moves), turns + 1)
                    player = players[index]
                    scores = []
                    for action in _legal_actions(node, player):
                        scores.append(choose(index + 1, moves | {player.id: action}))
                    return max(scores) if player.id == snake_id else min(scores)

                return choose(0, {})

            me = snakes_by_id[snake_id]
            try:
                action_scores: list[tuple[int, tuple[int, int, int, int]]] = []
                players = _living(state)
                for action in _legal_actions(state, me):
                    # Start with Devin's move so that he is the maximizing player.
                    moves = {snake_id: action}
                    remaining = tuple(snake for snake in players if snake.id != snake_id)

                    def choose_opponent(index: int, pending: dict[str, int]) -> tuple[int, int, int, int]:
                        nonlocal nodes
                        nodes += 1
                        if nodes > max_nodes:
                            raise _SearchLimitReached
                        if index == len(remaining):
                            return search_round(_advance(state, pending), 1)
                        opponent = remaining[index]
                        return min(
                            choose_opponent(index + 1, pending | {opponent.id: choice})
                            for choice in _legal_actions(state, opponent)
                        )

                    action_scores.append((action, choose_opponent(0, moves)))
                completed = max(action_scores, key=lambda item: item[1])
            except _SearchLimitReached:
                break

        return completed[0] if completed is not None else _one_ply_fallback(state, snake_id)

    return policy
