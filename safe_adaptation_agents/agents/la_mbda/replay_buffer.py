from typing import Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tf_agents.replay_buffers import episodic_replay_buffer

from safe_adaptation_agents import episodic_trajectory_buffer as etb
from safe_adaptation_agents.agents import Transition


class ReplayBuffer:

  def __init__(self, observation_shape: Tuple, action_shape: Tuple,
               max_length: int, seed: int, capacity: int, batch_size: int,
               sequence_length: int, precision: int):
    super(ReplayBuffer, self).__init__()
    dtype = {16: tf.float16, 32: tf.float32}[precision]
    self._sequence_length = sequence_length
    data_spec = {
        'observation': tf.TensorSpec(observation_shape, tf.uint8),
        'action': tf.TensorSpec(action_shape, dtype),
        'reward': tf.TensorSpec((), dtype),
        'cost': tf.TensorSpec((), dtype)
    }
    self._current_episode = {
        'observation': [],
        'action': [],
        'reward': [],
        'cost': [],
    }
    self._buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        data_spec,
        seed=seed,
        capacity=capacity,
        buffer_size=1,
        dataset_drop_remainder=True,
        completed_only=False,
        begin_episode_fn=lambda _: True,
        end_episode_fn=lambda _: True)
    self.idx = 0
    self._dtype = dtype
    self._dataset = self._buffer.as_dataset(batch_size,
                                            self._sequence_length + 1)
    self._dataset = self._dataset.map(self._preprocess,
                                      tf.data.experimental.AUTOTUNE)
    self._dataset = self._dataset.prefetch(10)

  def _preprocess(self, episode, _):
    episode['observation'] = preprocess(
        tf.cast(episode['observation'], self._dtype))
    # Shift observations, terminals and rewards by one timestep, since RSSM
    # always uses the *previous* action and state together with *current*
    # observation to infer the *current* state.
    episode['observation'] = episode['observation'][:, 1:]
    episode['cost'] = episode['cost'][:, 1:]
    episode['reward'] = episode['reward'][:, 1:]
    episode['action'] = episode['action'][:, :-1]
    return episode

  def store(self, transition: Transition):
    episode_end = transition.last
    self._current_episode['observation'].append(transition.observation[0])
    self._current_episode['action'].append(transition.action[0])
    self._current_episode['reward'].append(transition.reward[0])
    self._current_episode['cost'].append(transition.cost[0])
    if episode_end:
      self._current_episode['observation'].append(transition.next_observation)
      episode = {k: np.asarray(v) for k, v in self._current_episode.items()}
      new_idx = self._buffer.add_sequence(episode,
                                          tf.constant(self.idx, tf.int64))
      self.idx = int(new_idx)
      self._current_episode = {k: [] for k in self._current_episode.keys()}

  def sample(self, n_batches: int) -> Iterator[etb.TrajectoryData]:
    for batch in tfds.as_numpy(self._dataset.take(n_batches)):
      yield etb.TrajectoryData(batch['observation'], batch['action'],
                               batch['reward'], batch['cost'])

  def __getstate__(self):
    state = self.__dict__.copy()
    del state['_dataset']
    del state['_buffer']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)


def preprocess(image):
  return image / 255.0 - 0.5
