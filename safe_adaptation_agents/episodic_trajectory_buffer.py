from typing import Tuple, NamedTuple

import numpy as np

from safe_adaptation_agents.agents import Transition


class TrajectoryData(NamedTuple):
  o: np.ndarray
  a: np.ndarray
  r: np.ndarray
  c: np.ndarray


class EpisodicTrajectoryBuffer:

  def __init__(self,
               batch_size: int,
               max_length: int,
               observation_shape: Tuple,
               action_shape: Tuple,
               n_tasks: int = 1):
    self.idx = 0
    self.episode_id = 0
    self.task_id = 0
    self._full = False
    self.observation = np.zeros(
        (
            n_tasks,
            batch_size,
            max_length + 1,
        ) + observation_shape,
        dtype=np.float32)
    self.action = np.zeros(
        (
            n_tasks,
            batch_size,
            max_length,
        ) + action_shape, dtype=np.float32)
    self.reward = np.zeros((
        n_tasks,
        batch_size,
        max_length,
    ), dtype=np.float32)
    self.cost = np.zeros((
        n_tasks,
        batch_size,
        max_length,
    ), dtype=np.float32)

  def set_task(self, task_id: int):
    """
    Sets the current task id.
    """
    self.task_id = task_id
    self.episode_id = 0
    self.idx = 0

  def add(self, transition: Transition):
    """
    Adds transitions to the current running trajectory.
    """
    batch_size = min(transition.observation.shape[0], self.observation.shape[1])
    episode_slice = slice(self.episode_id, self.episode_id + batch_size)
    self.observation[self.task_id, episode_slice,
                     self.idx] = transition.observation[:batch_size].copy()
    self.action[self.task_id, episode_slice,
                self.idx] = transition.action[:batch_size].copy()
    self.reward[self.task_id, episode_slice,
                self.idx] = transition.reward[:batch_size].copy()
    self.cost[self.task_id, episode_slice,
              self.idx] = transition.cost[:batch_size].copy()
    if transition.last:
      assert self.idx == self.reward.shape[2] - 1
      self.observation[self.task_id, episode_slice, self.idx +
                       1] = transition.next_observation[:batch_size].copy()
      if self.episode_id + batch_size == self.observation.shape[
          1] and self.task_id + 1 == self.observation.shape[0]:
        self._full = True
      else:
        self.idx = -1
      self.episode_id += batch_size
    self.idx += 1

  def dump(self) -> TrajectoryData:
    """
    Returns all trajectories from all tasks (with shape [N_tasks, K_episodes,
    T_steps, ...]).
    """
    o = self.observation
    a = self.action
    r = self.reward
    c = self.cost
    # Reset the on-policy running cost.
    self.idx = 0
    self.episode_id = 0
    self.task_id = 0
    self._full = False
    if self.observation.shape[0] == 1:
      o, a, r, c = map(lambda x: x.squeeze(0), (o, a, r, c))
    return TrajectoryData(o, a, r, c)

  @property
  def full(self):
    return self._full
