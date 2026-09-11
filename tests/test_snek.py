import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from battlesnake_env import BoardState, Point, Snake
from snakes.snek import (
    EvaluationOptions,
    TrainingOptions,
    encode_state,
    evaluate,
    load_policy,
    network_action,
    train,
)
from snakes.snek.__main__ import _parser, main
from snakes.snek.training import _game_fitness, _match_seeds


def sample_state(*, food=None, opponents=True):
    if food is None:
        food = (Point(5, 3),)
    snakes = [Snake("learner", (Point(3, 3), Point(3, 2), Point(3, 1)), health=50)]
    if opponents:
        snakes.append(
            Snake("opponent", (Point(1, 3), Point(1, 2), Point(1, 1), Point(0, 1)))
        )
    return BoardState(7, 7, 25, food, (), tuple(snakes))


def test_encode_state_has_normalized_directional_features():
    features = encode_state(sample_state(), "learner", max_turns=100)

    assert features.shape == (20,)
    assert features.dtype == np.float32
    assert np.all(features >= -1.0)
    assert np.all(features <= 1.0)
    assert features[:4].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert features[4:8] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert features[12:16] == pytest.approx([2 / 6, 0.0, -2 / 6, 0.0])
    assert features[16:] == pytest.approx([0.5, 3 / 49, -1 / 49, 0.25])


def test_encode_state_uses_zero_vectors_when_targets_are_absent():
    features = encode_state(
        sample_state(food=(), opponents=False), "learner", max_turns=100
    )

    assert features[12:16].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert features[18] == 0.0


class StubNetwork:
    def __init__(self, outputs):
        self.outputs = outputs

    def activate(self, inputs):
        return self.outputs


def test_network_action_filters_unsafe_moves_and_breaks_ties_by_action():
    state = sample_state()

    assert (
        network_action(
            StubNetwork((0.8, 1.0, 0.8, 0.1)), state, "learner", max_turns=100
        )
        == 0
    )


def test_network_action_falls_back_when_no_action_is_advisory_safe():
    snake = Snake("learner", (Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)))
    state = BoardState(7, 7, 0, (), (), (snake,))

    assert (
        network_action(
            StubNetwork((0.0, 0.1, 0.9, 0.2)), state, "learner", max_turns=100
        )
        == 2
    )


@pytest.mark.parametrize(
    ("reward", "turns", "length", "expected"),
    [(1.0, 250, 20, 120.0), (-1.0, 125, 5, -91.0), (0.0, 250, 3, 10.0)],
)
def test_game_fitness(reward, turns, length, expected):
    assert _game_fitness(reward, turns, length, 250) == expected


def test_match_seeds_are_reproducible_and_generation_specific():
    assert _match_seeds(7, 2, 5) == _match_seeds(7, 2, 5)
    assert _match_seeds(7, 2, 5) != _match_seeds(7, 3, 5)


def test_training_saves_loadable_winner_and_statistics(tmp_path):
    result = train(
        TrainingOptions(
            generations=1,
            population=4,
            games=1,
            max_turns=2,
            seed=4,
            output_dir=tmp_path,
            checkpoint_interval=1,
        )
    )

    assert result.winner_path.parent.parent == tmp_path
    assert result.winner_path.exists()
    assert result.config_path.exists()
    assert result.statistics_path.exists()
    checkpoint = next(result.winner_path.parent.glob("checkpoint-*"))
    with result.statistics_path.open(newline="") as stream:
        assert next(csv.reader(stream)) == [
            "generation",
            "best_fitness",
            "mean_fitness",
        ]

    policy = load_policy(result.winner_path, max_turns=2)
    assert policy(sample_state(), "learner") in range(4)

    resumed = train(
        TrainingOptions(
            generations=1,
            population=4,
            games=1,
            max_turns=2,
            seed=4,
            output_dir=tmp_path / "resumed",
            checkpoint_interval=0,
            resume_checkpoint=checkpoint,
        )
    )
    assert resumed.winner_path.exists()


def test_training_is_reproducible(tmp_path):
    options = TrainingOptions(
        generations=1,
        population=4,
        games=1,
        max_turns=2,
        seed=8,
        checkpoint_interval=0,
        output_dir=tmp_path / "first",
    )

    first = train(options)
    second = train(replace(options, output_dir=tmp_path / "second"))

    assert first.fitness == second.fitness


def test_evaluate_reports_all_games():
    result = evaluate(
        lambda state, snake_id: 0, EvaluationOptions(games=2, max_turns=2, seed=3)
    )

    assert result.wins + result.ties + result.losses == 2
    assert result.average_turns > 0


def test_cli_rejects_non_positive_values():
    with pytest.raises(SystemExit):
        main(["train", "--games", "0"])


def test_training_cli_uses_train_snek_as_default_output_parent():
    assert _parser().parse_args(["train"]).output_dir == Path("train/snek")
