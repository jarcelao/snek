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

`snakes/sneklet.py` contains an example policy. Use `environment_policy` with
`BattlesnakeEnv` to play it in a local match.

```python
from battlesnake_env import BattlesnakeEnv, devious_devin
from snakes.sneklet import environment_policy

env = BattlesnakeEnv(opponent_policy=devious_devin(max_depth=2))
_, info = env.reset(seed=7)
action = environment_policy(info["state"], "snake-0")
```

Run a local match with a snake adapter. The default adapter is Sneklet:

```console
uv run python play.py
```

Select another adapter with `--snake`. Each adapter supplies its own options:

```console
uv run python play.py --snake snakes.snek --genome training/run-1/winner.pkl
```

Snake adapters must be modules inside `snakes` that define
`configure_play_parser(parser)` and `create_play_policy(args)`.

## NEAT snake

`snakes.snek` trains a feed-forward NEAT network against Devious Devin. The
network uses 20 compact inputs. They describe safe moves, nearby walls and
occupied cells, food, the nearest opponent, health, length, and turn progress.

Start a reproducible local training run:

```console
uv run python -m snakes.snek train
```

The command writes `winner.pkl`, the effective `config.ini`, a statistics CSV,
and periodic checkpoints to `snek-training/`. Use command options to change the
workload or output location:

```console
uv run python -m snakes.snek train \
  --generations 50 \
  --population 100 \
  --games 10 \
  --output-dir training/run-1
```

Resume a run from a checkpoint. `--generations` is the number of additional
generations to run:

```console
uv run python -m snakes.snek train \
  --resume-checkpoint training/run-1/checkpoint-10 \
  --generations 10 \
  --output-dir training/run-1
```

Evaluate a saved winner on a stable benchmark seed set:

```console
uv run python -m snakes.snek evaluate training/run-1/winner.pkl --games 50
```

Load the winner for use with `BattlesnakeEnv`:

```python
from battlesnake_env import BattlesnakeEnv, devious_devin
from snakes.snek import load_policy

policy = load_policy("training/run-1/winner.pkl")
env = BattlesnakeEnv(opponent_policy=devious_devin(max_depth=1))
_, info = env.reset(seed=7)
action = policy(info["state"], "snake-0")
```

Only load genome files that you trust. NEAT genome and checkpoint files use
Python pickle serialization.
