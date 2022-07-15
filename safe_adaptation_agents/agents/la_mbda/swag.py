from copy import deepcopy
from functools import partial
from typing import Union, Dict, Any, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax

from safe_adaptation_agents import utils as u


class SWAGLearningState(NamedTuple):
  learning_state: u.LearningState
  mu: hk.Params
  squared_mu: hk.Params
  covariance: hk.Params

  @property
  def params(self):
    return self.learning_state.params

  @property
  def opt_state(self):
    return self.learning_state.opt_state

  @property
  def iterations(self):
    return self.learning_state.opt_state[1][1].count


def cyclic_learning_rate(initial_value: float, peak_value: float,
                         cycle_steps: int):

  def schedule(count):
    cycle = jnp.floor(1 + count / (2. * cycle_steps))
    progress = jnp.abs(count / cycle_steps - 2 * cycle + 1)
    y = jnp.maximum(0, (1. - progress))
    return peak_value - (peak_value - initial_value) * y

  return schedule


class SWAG(u.Learner):

  def __init__(self,
               model: Union[hk.Transformed, hk.MultiTransformed,
                            chex.ArrayTree],
               seed: u.PRNGKey,
               optimizer_config: Dict,
               precision: jmp.Policy,
               *input_example: Any,
               start_averaging: int,
               average_period: int,
               max_num_models: int,
               learning_rate_factor: float,
               scale: float = 1.):
    super(SWAG, self).__init__(model, seed, optimizer_config, precision,
                               *input_example)
    base_lr = optimizer_config.get('lr', 1e-3)
    schedule = cyclic_learning_rate(base_lr, base_lr * learning_rate_factor,
                                    average_period)
    self.optimizer = optax.flatten(
        optax.chain(
            optax.clip_by_global_norm(
                optimizer_config.get('clip', float('inf'))),
            optax.adam(
                learning_rate=schedule, eps=optimizer_config.get('eps', 1e-8)),
        ))
    self.opt_state = self.optimizer.init(self.params)
    self._start_averaging = start_averaging
    self._average_period = average_period
    self._max_num_models = max_num_models
    self.scale = scale
    self.mu = jax.tree_map(jnp.zeros_like, self.params)
    self.squared_mu = deepcopy(self.mu)
    var = [
        jax.tree_map(lambda x: jnp.zeros_like(x).reshape(-1, 1), self.params)
        for _ in range(max_num_models)
    ]
    self.covariance_mat = u.pytrees_stack(var)

  @property
  def state(self):
    learning_state = super(SWAG, self).state
    return SWAGLearningState(learning_state, self.mu, self.squared_mu,
                             self.covariance_mat)

  @state.setter
  def state(self, state: SWAGLearningState):
    # Set Learner's `learning_state` in a hacky way.
    u.Learner.state.fset(self, state.learning_state)
    self.mu = state.mu
    self.squared_mu = state.squared_mu
    self.covariance_mat = state.covariance

  @property
  def warm(self):
    count = self.state.iterations
    average_period = self._average_period
    num_snapshots = max(0, (count - self._start_averaging) // average_period)
    max_num_models = self._max_num_models
    return count >= self._start_averaging and num_snapshots >= max_num_models

  def grad_step(self, grads, state: SWAGLearningState) -> SWAGLearningState:
    learning_state = super(SWAG, self).grad_step(grads, state.learning_state)
    mu, variance, covariance = self._update_stats(learning_state, state.mu,
                                                  state.squared_mu,
                                                  state.covariance)
    return SWAGLearningState(learning_state, mu, variance, covariance)

  def _update_stats(self, updated_state: u.LearningState, mu: hk.Params,
                    squared_mu: hk.Params,
                    covariance: hk.Params) -> [hk.Params, hk.Params, hk.Params]:
    # number of times snapshots of weights have been taken (using max to
    # avoid negative values of num_snapshots).
    iterations = updated_state.opt_state[1][1].count
    num_snapshots = jnp.maximum(0, (iterations - self._start_averaging) //
                                self._average_period)

    def compute_stats():

      def compute_mu(old_mean, value):
        new_mean = (old_mean * num_snapshots + value) / (num_snapshots + 1.)
        return new_mean

      def compute_var(old_sq_mu, value):
        new_sq_mu = (old_sq_mu * num_snapshots + value**2) / (
            num_snapshots + 1.)
        return new_sq_mu

      new_mu = jax.tree_map(compute_mu, mu, updated_state.params)
      new_sq_mu = jax.tree_map(compute_var, squared_mu, updated_state.params)

      def compute_cov(old_cov, value, new_mean):
        # Shift old covariances one step to the right. Update the leftmost
        # element with new covariance.
        old_cov = jnp.roll(old_cov, 1, 0)
        deviation = (value - new_mean).reshape(-1, 1)
        new_cov = old_cov.at[0].set(deviation)
        return new_cov

      new_cov = jax.tree_map(compute_cov, covariance, updated_state.params,
                             new_mu)
      return new_mu, new_sq_mu, new_cov

    # The mean update should happen iff two conditions are met:
    # 1. A min number of iterations (start_averaging) have taken place.
    # 2. Iteration is one in which snapshot should be taken.
    checkpoint = self._start_averaging + num_snapshots * self._average_period
    mu, squared_mu, covariance = jax.lax.cond(
        (iterations >= self._start_averaging) & (iterations == checkpoint),
        compute_stats, lambda: (mu, squared_mu, covariance))
    return mu, squared_mu, covariance

  def posterior_samples(self, num_samples: int, key: u.PRNGKey):
    state = self.state
    return _sample(
        state.mu,
        state.squared_mu,
        state.covariance,
        self.scale,
        jnp.asarray(jax.random.split(key, num_samples)),
    )


@jax.jit
@partial(jax.vmap, in_axes=(None, None, None, None, 0))
def _sample(mean: hk.Params, squared_mean: hk.Params, covariance: hk.Params,
            scale: float, key: u.PRNGKey) -> hk.Params:
  mean, unravel = jax.flatten_util.ravel_pytree(mean)
  squared_mean, _ = jax.flatten_util.ravel_pytree(squared_mean)
  variance = jnp.clip(squared_mean - mean**2, 1e-30)
  key, subkey = jax.random.split(key)
  variance_sample = jnp.sqrt(variance) * jax.random.normal(
      subkey, variance.shape)
  cov_mat_sqrt, _ = jax.tree_flatten(covariance)
  cov_mat_sqrt = jnp.concatenate(cov_mat_sqrt, 1)
  key, subkey = jax.random.split(key)
  max_num_models = cov_mat_sqrt.shape[0]
  covariance_sample = jnp.matmul(
      cov_mat_sqrt.T,
      jax.random.normal(subkey, (max_num_models,)),
  ).squeeze(0)
  covariance_sample /= jnp.sqrt(max_num_models - 1.)
  rand_sample = variance_sample + covariance_sample
  sample = mean + scale**0.5 * rand_sample
  sample = unravel(sample)
  return sample
