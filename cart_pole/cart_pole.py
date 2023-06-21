"""Cart Pole RL training and test code"""
import json
from collections.abc import Iterable, Iterator
from typing import Dict, List, NamedTuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE: int = 128
BATCH_SIZE: int = 16


class Net(nn.Module):
    """Feed-forward net to serve as RL agent"""
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int) -> None:
        """TODO:"""
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO:"""
        return self.net(x)


class EpisodeStep(NamedTuple):
    """TODO:"""
    observation: np.ndarray
    action: int


class Episode(NamedTuple):
    """Encapsulates an episode consisting of steps and rewards"""
    reward: float
    steps: List[EpisodeStep]


class Batch(NamedTuple):
    """TODO:"""
    observations: torch.FloatTensor
    actions: torch.LongTensor
    reward_bound: np.float64
    reward_mean: np.float64

    @property
    def json_accepted(
        self
    ) -> Dict[str, Union[List[float], np.float64]]:
        """Could use _asdict but tensors are not json saveable"""
        return {
            "observations": self.observations.tolist(),
            "actions": self.actions.tolist(),
            "reward_bound": self.reward_bound,
            "reward_mean": self.reward_mean,
        }


class EpisodeBatchGenerator:
    """TODO:"""
    # Batch of episodes

    def __init__(self, env, net, batch_size) -> None:
        """TODO:"""
        self._env = env
        self._net = net
        self._batch_size = batch_size

    def _generate_episode(self) -> Episode:
        """TODO:"""
        obs = self._env.reset()[0]
        sm = nn.Softmax(dim=0)
        is_done: bool = False
        episode_steps = []
        episode_reward = 0.0
        while not is_done:
            obs_v = torch.FloatTensor([obs])
            act_probs_v = sm(self._net(obs_v[0]))
            act_probs = act_probs_v.data.numpy()
            action = np.random.choice(len(act_probs), p=act_probs)

            next_obs, reward, is_done, *_ = env.step(action)
            episode_reward += reward
            episode_steps.append(EpisodeStep(observation=obs, action=action))
            obs = next_obs
        return Episode(reward=episode_reward, steps=episode_steps)

    def _filter_batch(
        self, batch: Iterable[Episode], percentile=70
    ) -> Batch:
        """TODO:"""
        rewards = tuple(item.reward for item in batch)
        reward_bound = np.percentile(rewards, percentile)
        # breakpoint()
        return Batch(
            observations=torch.FloatTensor(
                [
                    step.observation for reward, steps in batch for step in steps
                    if reward > reward_bound
                ]
            ),
            actions=torch.LongTensor(
                [
                    step.action for reward, steps in batch for step in steps
                    if reward > reward_bound
                ]
            ),
            reward_bound=reward_bound,
            reward_mean=np.mean(rewards),
        )

    def __iter__(self) -> Iterator[Batch]:
        """TODO:"""
        while True:
            yield self._filter_batch(
                tuple(self._generate_episode() for _ in range(self._batch_size))
            )


if __name__ == "__main__":
    """TODO: parameterize using click"""
    """
    https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/
    envs/registration.py#L509 marks the return of `gym.make` as `Env` whose
    observation_space is of type `Space` whose `.shape` returns type
    `Optional[Tuple[int, ...]]` as shown here https://github.com/openai/gym/blob/
    dcd185843a62953e27c2d54dc8c2d647d604b635/gym/spaces/space.py#L81

    This means `mypy` flags `env.observation_space.shape[0]` as an error since type
    `Optional[Tuple[int, ...]]` cannot be indexed

    Additionally, `action_space` of `env` is of type `Space` meaning it does not have an
    `n` attribute
    """
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]  # type: ignore
    n_actions = env.action_space.n  # type: ignore

    net: Net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(params=net.parameters(), lr=0.01)

    trials: List[Dict[str, Union[List[float], np.float64]]] = []
    for iter_no, batch in enumerate(
        EpisodeBatchGenerator(env=env, net=net, batch_size=BATCH_SIZE)
    ):
        optimizer.zero_grad()
        loss_v = objective(net(batch.observations), batch.actions)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), batch.reward_mean, batch.reward_bound))
        trials.append(batch.json_accepted)
        if batch.reward_mean > 199:
            print("Solved!")
            break
    with open("test.json", "w") as f:
        json.dump(trials, f)

