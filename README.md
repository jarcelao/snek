# A local Battlesnake simulation environment.

## Training environment

`battlesnake_env.BattlesnakeEnv` is a single-agent Gymnasium environment for
the standard Battlesnake ruleset. The learning snake uses actions `0` through
`3` for up, down, left, and right. Supply one opponent callable for all other
snakes.

```python
from battlesnake_env import BattlesnakeEnv

env = BattlesnakeEnv(opponent_policy=lambda state, snake_id: 0)
observation, info = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(0)
```

`devious_devin()` provides a deterministic paranoid-minimax opponent. Its
`max_depth` value is the number of complete simultaneous turns it searches.

```python
from battlesnake_env import BattlesnakeEnv, devious_devin

env = BattlesnakeEnv(opponent_policy=devious_devin(max_depth=2))
```

The observation is an eight-channel `float32` board tensor. `info` contains an
immutable `BoardState` and an advisory action mask. Use `render_mode="ansi"`
to return a text board for debugging.

## Snake policy

`snek.py` contains an example policy. Use `environment_policy` with
`BattlesnakeEnv` to play it in a local match.

```python
from battlesnake_env import BattlesnakeEnv, devious_devin
from snek import environment_policy

env = BattlesnakeEnv(opponent_policy=devious_devin(max_depth=2))
_, info = env.reset(seed=7)
action = environment_policy(info["state"], "snake-0")
```
