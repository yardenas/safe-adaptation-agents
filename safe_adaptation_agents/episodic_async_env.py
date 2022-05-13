import sys
import functools
from typing import Optional, Union, List, Callable

import atexit
import traceback

import cloudpickle

import multiprocessing as mp

from enum import Enum

import numpy as np

from gym.vector import VectorEnv
from gym import Env


class Protocol(Enum):
  ACCESS = 0
  CALL = 1
  RESULT = 2
  EXCEPTION = 3
  CLOSE = 4


# Based on https://github.com/danijar/dreamerv2/blob
# /07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/common/envs.py#L522 as
# OpenAI gym's AsynVectorEnv fails to render nicely together with dm-control.
# (The main issue is with creating a dm-control
# https://github.com/openai/gym/blob/9a5db3b77a0c880ffed96ece1ab76eeff92c85e1
# /gym/vector/async_vector_env.py#L127 which loads all the rendering handler
# in the main process.)
class EpisodicAsync(VectorEnv):

  def __init__(self, ctor: Callable[[], Env], vector_size: int = 1):
    self.env_fn = cloudpickle.dumps(ctor)

    if vector_size < 1:
      self._env = ctor()
      self.observation_space = self._env.observation_space
      self.action_space = self._env.action_space
    else:
      self._env = None
      self.parents, self.processes = zip(
          *[self._make_worker() for _ in range(vector_size)])
      atexit.register(self.close)
      for process in self.processes:
        process.start()
      self.observation_space = self.get_attr('observation_space')[0]
      self.action_space = self.get_attr('action_space')[0]

  def _make_worker(self):
    parent, child = mp.Pipe()
    process = mp.Process(target=_worker, args=(self.env_fn, child))
    return parent, process

  @functools.lru_cache
  def get_attr(self, name):
    if self._env is not None:
      return getattr(self._env, name)
    for parent in self.parents:
      parent.send((Protocol.ACCESS, name))
    return self._receive()

  def close(self):
    if self._env is not None:
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      for parent in self.parents:
        parent.send((Protocol.CLOSE, None))
        parent.close()
    except IOError:
      # The connection was already closed.
      pass
    for process in self.processes:
      process.join()

  def _receive(self):
    payloads = []
    for parent in self.parents:
      try:
        message, payload = parent.recv()
      except ConnectionResetError:
        raise RuntimeError('Environment worker crashed.')
      # Re-raise exceptions in the main process.
      if message == Protocol.EXCEPTION:
        stacktrace = payload
        raise Exception(stacktrace)
      if message == Protocol.RESULT:
        payloads.append(payload)
      else:
        raise KeyError(f'Received message of unexpected type {message}')
    assert len(payloads) == len(self.parents)
    return payloads

  def step_async(self, actions):
    for parent, action in zip(self.parents, actions):
      payload = 'step', (action,), {}
      parent.send((Protocol.CALL, payload))

  def step_wait(self, **kwargs):
    observations, rewards, dones, infos = zip(*self._receive())
    if any(dones):
      assert all(dones), (
          'We treat only the episodic case, so done means that time limit was '
          'reached')
      for info in infos:
        info['last_observation'] = observations
      observations = self.call('reset')
    return np.asarray(observations), np.asarray(rewards), np.asarray(
        dones, dtype=bool), infos

  def call_async(self, name, *args, **kwargs):
    if self._env is not None:
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    for parent in self.parents:
      parent.send((Protocol.CALL, payload))

  def call_wait(self, **kwargs):
    return self._receive()

  def set_attr(self, name, values):
    pass

  def render(self, mode="human"):
    self.call_async('render', mode)
    return np.asarray(self.call_wait())

  def reset(self,
            seed: Optional[Union[int, List[int]]] = None,
            return_info: bool = False,
            options: Optional[dict] = None):
    return self.reset_wait(seed, return_info, options)

  def reset_wait(self,
                 seed: Optional[Union[int, List[int]]] = None,
                 return_info: bool = False,
                 options: Optional[dict] = None):
    self.call_async(
        'reset', seed=seed, return_info=return_info, options=options)
    return np.asarray(self.call_wait())


def _worker(ctor, conn):
  try:
    env = cloudpickle.loads(ctor)()
    while True:
      try:
        # Only block for short times to have keyboard exceptions be raised.
        if not conn.poll(0.1):
          continue
        message, payload = conn.recv()
      except (EOFError, KeyboardInterrupt):
        break
      if message == Protocol.ACCESS:
        name = payload
        result = getattr(env, name)
        conn.send((Protocol.RESULT, result))
        continue
      if message == Protocol.CALL:
        name, args, kwargs = payload
        result = getattr(env, name)(*args, **kwargs)
        conn.send((Protocol.RESULT, result))
        continue
      if message == Protocol.CLOSE:
        assert payload is None
        break
      raise KeyError(f'Received message of unknown type {message}')
  except Exception:
    stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
    print(f'Error in environment process: {stacktrace}')
    conn.send((Protocol.EXCEPTION, stacktrace))
  finally:
    env.close()  # noqa
    conn.close()