"""Gymnasium environment for the standard Battlesnake ruleset."""

from __future__ import annotations

from collections.abc import Callable
import random
from typing import TypeAlias

import gymnasium as gym
import numpy as np

from .rules import BoardState, DELTAS, MAX_HEALTH, Settings, initial_state, step

OpponentPolicy: TypeAlias = Callable[[BoardState, str], int]


class BattlesnakeEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, opponent_policy: OpponentPolicy, *, width: int = 11, height: int = 11,
                 num_snakes: int = 2, food_spawn_chance: int = 15, minimum_food: int = 1,
                 hazard_damage_per_turn: int = 14, max_turns: int | None = 500,
                 render_mode: str | None = None, ruleset: str = "standard", game_map: str = "standard"):
        if ruleset != "standard" or game_map != "standard":
            raise ValueError("only the standard ruleset and standard map are supported")
        if width != height or width < 7 or width > 25 or width % 2 == 0:
            raise ValueError("standard boards must be odd squares from 7 through 25")
        if not 2 <= num_snakes <= 16:
            raise ValueError("num_snakes must be from 2 through 16")
        if num_snakes > 8 and width < 11:
            raise ValueError("standard boards below 11x11 support at most eight snakes")
        if not callable(opponent_policy):
            raise TypeError("opponent_policy must be callable")
        if not 0 <= food_spawn_chance <= 100 or minimum_food < 0 or hazard_damage_per_turn < 0:
            raise ValueError("settings must be non-negative and food_spawn_chance must be at most 100")
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be positive or None")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.opponent_policy, self.width, self.height, self.num_snakes = opponent_policy, width, height, num_snakes
        self.settings, self.max_turns, self.render_mode = Settings(food_spawn_chance, minimum_food, hazard_damage_per_turn), max_turns, render_mode
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(8, height, width), dtype=np.float32)
        self.state: BoardState | None = None
        self._rng = random.Random()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = random.Random(seed)
        self.state = initial_state(self.width, self.height, tuple(f"snake-{i}" for i in range(self.num_snakes)), self._rng)
        return self._observation(), self._info()

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer from 0 through 3")
        learner = self.state.snakes[0]
        if not learner.alive:
            raise RuntimeError("episode is complete; call reset")
        moves = {learner.id: int(action)}
        for snake in self.state.snakes[1:]:
            if snake.alive:
                choice = self.opponent_policy(self.state, snake.id)
                if not self.action_space.contains(choice):
                    raise ValueError("opponent_policy must return an integer from 0 through 3")
                moves[snake.id] = int(choice)
        self.state, already_over = step(self.state, moves, self.settings, self._rng)
        alive = [snake for snake in self.state.snakes if snake.alive]
        terminated = already_over or len(alive) <= 1
        truncated = not terminated and self.max_turns is not None and self.state.turn >= self.max_turns
        reward = 0.0
        if terminated:
            if len(alive) == 1 and alive[0].id == learner.id:
                reward = 1.0
            elif not alive:
                reward = 0.0
            elif not any(s.id == learner.id for s in alive):
                reward = -1.0
        return self._observation(), reward, terminated, truncated, self._info()

    def _observation(self) -> np.ndarray:
        assert self.state is not None
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        learner = self.state.snakes[0]
        for snake in self.state.snakes:
            if not snake.alive:
                continue
            head_plane, body_plane = (0, 1) if snake.id == learner.id else (2, 3)
            if 0 <= snake.head.x < self.width and 0 <= snake.head.y < self.height:
                obs[head_plane, snake.head.y, snake.head.x] = 1.0
            for point in snake.body[1:]:
                if 0 <= point.x < self.width and 0 <= point.y < self.height:
                    obs[body_plane, point.y, point.x] = 1.0
        for point in self.state.food:
            obs[4, point.y, point.x] = 1.0
        for point in self.state.hazards:
            obs[5, point.y, point.x] = 1.0
        obs[6].fill(learner.health / MAX_HEALTH if learner.alive else 0.0)
        obs[7].fill(0.0 if self.max_turns is None else min(1.0, self.state.turn / self.max_turns))
        return obs

    def _info(self) -> dict:
        assert self.state is not None
        learner = self.state.snakes[0]
        mask = np.zeros(4, dtype=np.int8)
        if learner.alive:
            for action, (dx, dy) in DELTAS.items():
                point = type(learner.head)(learner.head.x + dx, learner.head.y + dy)
                mask[action] = int(0 <= point.x < self.width and 0 <= point.y < self.height and point not in learner.body[1:])
        return {"action_mask": mask, "state": self.state}

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        assert self.state is not None
        cells = [["." for _ in range(self.width)] for _ in range(self.height)]
        for point in self.state.food:
            cells[point.y][point.x] = "F"
        for index, snake in enumerate(self.state.snakes):
            if not snake.alive:
                continue
            for point in snake.body[1:]:
                if 0 <= point.x < self.width and 0 <= point.y < self.height:
                    cells[point.y][point.x] = str(index)
            if 0 <= snake.head.x < self.width and 0 <= snake.head.y < self.height:
                cells[snake.head.y][snake.head.x] = "A" if index == 0 else str(index)
        return "\n".join("".join(row) for row in reversed(cells))
