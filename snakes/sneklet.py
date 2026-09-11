"""
Basic snake adapted from the official Battlesnake Python template:
https://github.com/BattlesnakeOfficial/starter-snake-python
"""

import argparse

from battlesnake_env import OpponentPolicy
from battlesnake_env.rules import BoardState

ACTION_BY_MOVE = {"up": 0, "down": 1, "left": 2, "right": 3}


def move(game_state: dict) -> dict:
    """Select a move from a Battlesnake-format game state."""
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    my_head = game_state["you"]["body"][0]
    my_neck = game_state["you"]["body"][1]

    if my_neck["x"] < my_head["x"]:
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:
        is_move_safe["up"] = False

    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    occupied = {
        (part["x"], part["y"])
        for snake in game_state["board"]["snakes"]
        for part in snake["body"][:-1]
    }
    deltas = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    safe_moves = [
        direction
        for direction, is_safe in is_move_safe.items()
        if is_safe
        and 0 <= my_head["x"] + deltas[direction][0] < board_width
        and 0 <= my_head["y"] + deltas[direction][1] < board_height
        and (my_head["x"] + deltas[direction][0], my_head["y"] + deltas[direction][1])
        not in occupied
    ]

    if not safe_moves:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    food = game_state["board"]["food"]

    def food_distance(direction: str) -> int:
        dx, dy = deltas[direction]
        next_head = (my_head["x"] + dx, my_head["y"] + dy)
        return min(
            (
                abs(next_head[0] - item["x"]) + abs(next_head[1] - item["y"])
                for item in food
            ),
            default=0,
        )

    next_move = min(safe_moves, key=food_distance)
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


def environment_policy(state: BoardState, snake_id: str) -> int:
    """Select an action for a snake in ``BattlesnakeEnv``."""
    snake = next(snake for snake in state.snakes if snake.id == snake_id)
    game_state = {
        "turn": state.turn,
        "you": {
            "id": snake.id,
            "health": snake.health,
            "body": [{"x": part.x, "y": part.y} for part in snake.body],
        },
        "board": {
            "width": state.width,
            "height": state.height,
            "food": [{"x": food.x, "y": food.y} for food in state.food],
            "snakes": [
                {
                    "id": other.id,
                    "health": other.health,
                    "body": [{"x": part.x, "y": part.y} for part in other.body],
                }
                for other in state.snakes
                if other.alive
            ],
        },
    }
    return ACTION_BY_MOVE[move(game_state)["move"]]


def configure_play_parser(parser: argparse.ArgumentParser) -> None:
    """Add Sneklet options to the local match parser."""


def create_play_policy(args: argparse.Namespace) -> OpponentPolicy:
    """Create Sneklet's environment policy for a local match."""
    return environment_policy
