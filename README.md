# A Battlesnake service written in Python.

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

The observation is an eight-channel `float32` board tensor. `info` contains an
immutable `BoardState` and an advisory action mask. Use `render_mode="ansi"`
to return a text board for debugging.
