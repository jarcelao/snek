from battlesnake_env import BattlesnakeEnv, devious_devin
from main import environment_policy


def test_template_snake_can_play_against_devious_devin():
    env = BattlesnakeEnv(devious_devin(max_depth=1), max_turns=100)
    _, info = env.reset(seed=7)

    for _ in range(100):
        action = environment_policy(info["state"], "snake-0")
        assert env.action_space.contains(action)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    else:
        raise AssertionError("the match did not finish within the turn limit")
