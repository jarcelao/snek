import pytest

from battlesnake_env import BoardState, Point, Snake, devious_devin
from battlesnake_env.devious_devin import _FROZEN_FOOD_SETTINGS, _advance, _score
from battlesnake_env.rules import DOWN, LEFT, RIGHT, UP, Elimination


def state_for_devin(*, me_body, opponent_body, food=(), me_health=100):
    return BoardState(
        7,
        7,
        0,
        tuple(food),
        (),
        (
            Snake("devin", tuple(me_body), me_health),
            Snake("other", tuple(opponent_body)),
        ),
    )


def test_factory_returns_deterministic_valid_policy():
    state = state_for_devin(
        me_body=(Point(3, 3), Point(3, 2)),
        opponent_body=(Point(5, 5), Point(5, 4)),
        food=(Point(3, 5),),
    )
    policy = devious_devin()

    assert policy(state, "devin") == policy(state, "devin")
    assert policy(state, "devin") in (UP, LEFT, RIGHT)


@pytest.mark.parametrize(
    "kwargs", ({"max_depth": 0}, {"max_nodes": 0}, {"max_depth": True})
)
def test_factory_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        devious_devin(**kwargs)


def test_score_prefers_food_when_equal_or_hungry():
    equal = state_for_devin(
        me_body=(Point(1, 1), Point(1, 0), Point(0, 0)),
        opponent_body=(Point(5, 5), Point(5, 4), Point(5, 3)),
        food=(Point(2, 1),),
    )
    hungry = state_for_devin(
        me_body=(Point(1, 1), Point(1, 0), Point(0, 0), Point(0, 1)),
        opponent_body=(Point(5, 5), Point(5, 4), Point(5, 3)),
        food=(Point(2, 1),),
        me_health=19,
    )

    assert _score(equal, "devin", 0)[0] == 2
    assert _score(hungry, "devin", 0)[0] == 2


def test_score_prefers_opponent_when_longer_and_healthy():
    state = state_for_devin(
        me_body=(Point(1, 1), Point(1, 0), Point(0, 0), Point(0, 1)),
        opponent_body=(Point(5, 5), Point(5, 4), Point(5, 3)),
        food=(Point(2, 1),),
    )

    assert _score(state, "devin", 0)[0] == 3


def test_score_orders_terminal_results():
    winner = BoardState(
        7,
        7,
        1,
        (),
        (),
        (
            Snake("devin", (Point(1, 1),)),
            Snake("other", (Point(5, 5),), elimination=Elimination.OUT_OF_BOUNDS),
        ),
    )
    tie = BoardState(
        7,
        7,
        1,
        (),
        (),
        (
            Snake("devin", (Point(1, 1),), elimination=Elimination.OUT_OF_BOUNDS),
            Snake("other", (Point(5, 5),), elimination=Elimination.OUT_OF_BOUNDS),
        ),
    )
    loss = BoardState(
        7,
        7,
        1,
        (),
        (),
        (
            Snake("devin", (Point(1, 1),), elimination=Elimination.OUT_OF_BOUNDS),
            Snake("other", (Point(5, 5),)),
        ),
    )

    assert (
        _score(winner, "devin", 1) > _score(tie, "devin", 1) > _score(loss, "devin", 1)
    )


def test_projection_freezes_future_food():
    state = state_for_devin(
        me_body=(Point(1, 1), Point(1, 0)),
        opponent_body=(Point(5, 5), Point(5, 4)),
    )

    next_state = _advance(state, {"devin": UP, "other": DOWN})

    assert next_state.food == ()
    assert _FROZEN_FOOD_SETTINGS.food_spawn_chance == 0


def test_node_limit_uses_a_valid_deterministic_fallback():
    state = state_for_devin(
        me_body=(Point(3, 3), Point(3, 2)),
        opponent_body=(Point(5, 5), Point(5, 4)),
        food=(Point(3, 5),),
    )
    policy = devious_devin(max_depth=2, max_nodes=1)

    assert policy(state, "devin") == policy(state, "devin")
    assert policy(state, "devin") in (UP, LEFT, RIGHT)
