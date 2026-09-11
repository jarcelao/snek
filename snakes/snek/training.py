"""Training and evaluation support for the NEAT snake."""

from __future__ import annotations

import csv
import pickle
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import neat

from battlesnake_env import BattlesnakeEnv, BoardState, OpponentPolicy, devious_devin

from .policy import Network, _load_config, network_action


class _NeatConfig(Protocol):
    pop_size: int
    seed: int | None

    def save(self, filename: str) -> None: ...


@dataclass(frozen=True)
class TrainingOptions:
    """Settings for one deterministic training run."""

    generations: int = 20
    population: int = 50
    games: int = 5
    max_turns: int = 250
    devin_depth: int = 1
    seed: int = 0
    output_dir: Path = Path("train/snek")
    checkpoint_interval: int = 5
    resume_checkpoint: Path | None = None


@dataclass(frozen=True)
class EvaluationOptions:
    """Settings for policy evaluation."""

    games: int = 20
    max_turns: int = 250
    devin_depth: int = 1
    seed: int = 10_000


@dataclass(frozen=True)
class TrainingResult:
    """Files and final score from a training run."""

    winner_path: Path
    config_path: Path
    statistics_path: Path
    fitness: float


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate match results."""

    wins: int
    ties: int
    losses: int
    average_fitness: float
    average_turns: float


def _network_policy(network: Network, max_turns: int) -> OpponentPolicy:
    """Adapt a NEAT network to the environment's policy protocol."""

    def policy(state: BoardState, snake_id: str) -> int:
        return network_action(network, state, snake_id, max_turns=max_turns)

    return policy


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _game_fitness(
    reward: float, turns: int, final_length: int, max_turns: int
) -> float:
    """Calculate outcome-first shaped fitness for one match."""
    outcome = 100.0 if reward > 0 else -100.0 if reward < 0 else 0.0
    survival = 10.0 * min(turns, max_turns) / max_turns
    growth = min(10.0, max(0, final_length - 3) * 2.0)
    return outcome + survival + growth


def _play(
    policy: OpponentPolicy, options: EvaluationOptions, match_seed: int
) -> tuple[float, float, int]:
    env = BattlesnakeEnv(
        devious_devin(max_depth=options.devin_depth),
        max_turns=options.max_turns,
    )
    _, info = env.reset(seed=match_seed)
    while True:
        action = policy(info["state"], "snake-0")
        _, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            snake = info["state"].snakes[0]
            score = _game_fitness(
                reward, info["state"].turn, len(snake.body), options.max_turns
            )
            return reward, score, info["state"].turn


def _match_seeds(seed: int, generation: int, games: int) -> tuple[int, ...]:
    rng = random.Random((seed << 32) ^ generation)
    return tuple(rng.randrange(2**31) for _ in range(games))


def _create_run_output_dir(parent_dir: Path) -> Path:
    """Create a unique timestamped directory for one training run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = parent_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def evaluate(
    policy: OpponentPolicy, options: EvaluationOptions | None = None
) -> EvaluationResult:
    """Evaluate a policy on a stable benchmark seed set."""
    effective_options = EvaluationOptions() if options is None else options
    _validate_positive("games", effective_options.games)
    _validate_positive("max_turns", effective_options.max_turns)
    _validate_positive("devin_depth", effective_options.devin_depth)
    outcomes: list[float] = []
    scores: list[float] = []
    turns: list[int] = []
    for match_seed in _match_seeds(effective_options.seed, 0, effective_options.games):
        reward, score, turn = _play(policy, effective_options, match_seed)
        outcomes.append(reward)
        scores.append(score)
        turns.append(turn)
    return EvaluationResult(
        wins=sum(reward > 0 for reward in outcomes),
        ties=sum(reward == 0 for reward in outcomes),
        losses=sum(reward < 0 for reward in outcomes),
        average_fitness=sum(scores) / len(scores),
        average_turns=sum(turns) / len(turns),
    )


class _GenomeEvaluator:
    def __init__(self, options: TrainingOptions, generation: int):
        self.options = options
        self.generation = generation

    def __call__(self, genomes, config) -> None:
        seeds = _match_seeds(self.options.seed, self.generation, self.options.games)
        match_options = EvaluationOptions(
            games=self.options.games,
            max_turns=self.options.max_turns,
            devin_depth=self.options.devin_depth,
            seed=self.options.seed,
        )
        for _, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            policy = _network_policy(network, self.options.max_turns)

            genome.fitness = sum(
                _play(policy, match_options, seed)[1] for seed in seeds
            ) / len(seeds)
        self.generation += 1


def train(options: TrainingOptions | None = None) -> TrainingResult:
    """Train a population and save its winner and statistics."""
    effective_options = TrainingOptions() if options is None else options
    for name in ("generations", "population", "games", "max_turns", "devin_depth"):
        _validate_positive(name, getattr(effective_options, name))
    if effective_options.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")

    output_dir = _create_run_output_dir(Path(effective_options.output_dir))
    config_path = output_dir / "config.ini"
    base_config = Path(__file__).with_name("config.ini")
    config = cast(_NeatConfig, _load_config(base_config))
    config.pop_size = effective_options.population
    config.seed = effective_options.seed
    config.save(str(config_path))

    if effective_options.resume_checkpoint is None:
        population = neat.Population(config)
    else:
        population = neat.Checkpointer.restore_checkpoint(
            str(effective_options.resume_checkpoint), new_config=config
        )
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    population.add_reporter(neat.StdOutReporter(False))
    if effective_options.checkpoint_interval:
        population.add_reporter(
            neat.Checkpointer(
                effective_options.checkpoint_interval,
                filename_prefix=str(output_dir / "checkpoint-"),
            )
        )

    evaluator = _GenomeEvaluator(effective_options, population.generation)
    winner = population.run(evaluator, effective_options.generations)
    winner_path = output_dir / "winner.pkl"
    with winner_path.open("wb") as stream:
        pickle.dump(winner, stream, protocol=pickle.HIGHEST_PROTOCOL)

    statistics_path = output_dir / "statistics.csv"
    means = statistics.get_fitness_mean()
    with statistics_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("generation", "best_fitness", "mean_fitness"))
        for generation, (genome, mean) in enumerate(
            zip(statistics.most_fit_genomes, means)
        ):
            writer.writerow(
                (population.generation - len(means) + generation, genome.fitness, mean)
            )
    return TrainingResult(
        winner_path, config_path, statistics_path, float(winner.fitness)
    )
