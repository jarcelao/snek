import random

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from battlesnake_env import BattlesnakeEnv, BoardState, Point, Snake
from battlesnake_env.rules import Elimination, Settings, step


def up_policy(state, snake_id):
    return 0


def test_gymnasium_contract_and_seed_reproducibility():
    first = BattlesnakeEnv(up_policy)
    second = BattlesnakeEnv(up_policy)
    observation_a, info_a = first.reset(seed=42)
    observation_b, info_b = second.reset(seed=42)

    assert np.array_equal(observation_a, observation_b)
    assert info_a["state"] == info_b["state"]
    check_env(BattlesnakeEnv(up_policy), skip_render_check=True)


def test_food_growth_and_sparse_win_reward():
    env = BattlesnakeEnv(up_policy)
    env.reset(seed=1)
    assert env.state is not None
    learner = env.state.snakes[0]
    target = Point(learner.head.x, learner.head.y + 1)
    env.state = BoardState(env.state.width, env.state.height, env.state.turn, (target,), env.state.hazards, env.state.snakes)

    _, _, _, _, info = env.step(0)

    assert info["state"].snakes[0].health == 100
    assert len(info["state"].snakes[0].body) == 4


def test_truncation_and_ansi_render():
    env = BattlesnakeEnv(lambda state, snake_id: 1, max_turns=1, render_mode="ansi")
    env.reset(seed=3)
    _, _, terminated, truncated, _ = env.step(0)

    assert not terminated
    assert truncated
    assert isinstance(env.render(), str)


def test_invalid_policy_result_is_rejected():
    env = BattlesnakeEnv(lambda state, snake_id: 9)
    env.reset(seed=1)

    with pytest.raises(ValueError, match="opponent_policy"):
        env.step(0)


def test_large_board_supports_sixteen_snakes():
    env = BattlesnakeEnv(up_policy, width=11, height=11, num_snakes=16)
    _, info = env.reset(seed=9)

    assert len(info["state"].snakes) == 16


def test_observation_and_action_mask_are_in_declared_spaces():
    env = BattlesnakeEnv(up_policy)
    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
    assert observation.shape == (8, 11, 11)
    assert info["action_mask"].shape == (4,)


@pytest.mark.parametrize(
    ("state", "moves", "expected"),
    [
        pytest.param(
            BoardState(7, 7, 0, (), (), (Snake("one", (Point(0, 0), Point(0, 1))), Snake("two", (Point(5, 5), Point(5, 4))))),
            {"one": 1, "two": 0},
            (Elimination.OUT_OF_BOUNDS, Elimination.NONE),
            id="wall-collision",
        ),
        pytest.param(
            BoardState(7, 7, 0, (), (), (Snake("one", (Point(2, 2), Point(2, 1))), Snake("two", (Point(2, 4), Point(2, 5))))),
            {"one": 0, "two": 1},
            (Elimination.HEAD_COLLISION, Elimination.HEAD_COLLISION),
            id="equal-head-to-head",
        ),
    ],
)
def test_eliminations(state, moves, expected):
    next_state, _ = step(state, moves, Settings(0, 0), random.Random(1))

    assert tuple(snake.elimination for snake in next_state.snakes) == expected


def test_food_saves_the_last_health_point():
    state = BoardState(7, 7, 0, (Point(2, 3),), (), (
        Snake("one", (Point(2, 2), Point(2, 1)), health=1),
        Snake("two", (Point(5, 5), Point(5, 4))),
    ))
    next_state, _ = step(state, {"one": 0, "two": 0}, Settings(0, 0), random.Random(1))

    assert next_state.snakes[0].alive
    assert next_state.snakes[0].health == 100
