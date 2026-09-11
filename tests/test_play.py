import argparse

import pytest

import play
from snakes import snek, sneklet


def test_play_defaults_to_sneklet_and_finishes_a_seeded_match(capsys):
    reward, turn, result = play.play(7, max_turns=100)

    assert reward in (-1.0, 0.0, 1.0)
    assert 0 < turn <= 100
    assert result in {"won", "lost", "tied or reached the turn limit"}
    assert "Sneklet" in capsys.readouterr().out


def test_sneklet_adapter_adds_no_options_and_returns_its_policy():
    parser = argparse.ArgumentParser()
    sneklet.configure_play_parser(parser)

    assert (
        sneklet.create_play_policy(parser.parse_args([])) is sneklet.environment_policy
    )


def test_snek_adapter_loads_the_selected_model(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, default=500)
    snek.configure_play_parser(parser)
    args = parser.parse_args(
        [
            "--genome",
            str(tmp_path / "winner.pkl"),
            "--config",
            str(tmp_path / "config.ini"),
            "--max-turns",
            "75",
        ]
    )
    policy = lambda state, snake_id: 0
    calls = []

    def load(genome, config, *, max_turns):
        calls.append((genome, config, max_turns))
        return policy

    monkeypatch.setattr(snek, "load_policy", load)

    assert snek.create_play_policy(args) is policy
    assert calls == [(tmp_path / "winner.pkl", tmp_path / "config.ini", 75)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--snake", "not_snakes"],
        ["--snake", "snakes.snek.policy"],
    ],
)
def test_main_rejects_invalid_snake_adapters(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        play.main(arguments)

    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "snake adapter" in output or "snakes package" in output
