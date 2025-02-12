from __future__ import annotations

import gymnasium.spaces
import numpy as np

from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv


class BaseParallelWrapper(ParallelEnv):
    def __init__(self, env: ParallelEnv):
        print("init baseparallelwrapper")
        super().__init__()
        self.env = env

    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        return self.env.reset(seed=seed, options=options)

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        if self.env.unwrapped.env.terminate:
            print("terminated")
            if self.env.unwrapped.env.ball.rect.center[0] < self.env.unwrapped.env.area.center[0]:
                print('left paddle failed')
                rewards['paddle_0'] = -self.env.unwrapped.env.off_screen_penalty
                rewards['paddle_1'] = self.env.unwrapped.env.off_screen_penalty
            else:
                print('right paddle failed')
                rewards['paddle_0'] = self.env.unwrapped.env.off_screen_penalty
                rewards['paddle_1'] = -self.env.unwrapped.env.off_screen_penalty

        # print("wrap", rewards, terminations, truncations)

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def close(self) -> None:
        return self.env.close()

    @property
    def unwrapped(self) -> ParallelEnv:
        return self.env.unwrapped

    def state(self) -> np.ndarray:
        return self.env.state()

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)