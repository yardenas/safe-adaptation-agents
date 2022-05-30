from typing import Tuple, Optional
from types import SimpleNamespace
import functools

import numpy as np
from gym.spaces import Space

import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import optax

from safe_adaptation_agents.agents.agent import Transition
from safe_adaptation_agents.agents.on_policy import ppo_lagrangian, vpg
from safe_adaptation_agents.agents.on_policy.safe_vpg import Evaluation
from safe_adaptation_agents.logging import TrainingLogger
from safe_adaptation_agents.utils import LearningState
from safe_adaptation_agents import utils
from safe_adaptation_agents import episodic_trajectory_buffer as etb
from safe_adaptation_agents.episodic_trajectory_buffer import TrajectoryData

# Vmap over the task axis.
discounted_cumsum = jax.vmap(vpg.discounted_cumsum, in_axes=[0, None])


class MamlPpoLagrangian(ppo_lagrangian.PpoLagrangian):

  def __init__(self, observation_space: Space, action_space: Space,
               config: SimpleNamespace, logger: TrainingLogger,
               actor: hk.Transformed, critic: hk.Transformed,
               safety_critic: hk.Transformed):
    super(MamlPpoLagrangian,
          self).__init__(observation_space, action_space, config, logger, actor,
                         critic, safety_critic)
    num_steps = self.config.time_limit // self.config.action_repeat
    self.buffer = etb.EpisodicTrajectoryBuffer(self.config.num_trajectories,
                                               num_steps,
                                               observation_space.shape,
                                               action_space.shape,
                                               self.config.task_batch_size)
    self.query_buffer = etb.EpisodicTrajectoryBuffer(
        self.config.num_query_trajectories, num_steps, observation_space.shape,
        action_space.shape, self.config.task_batch_size)
    self.task_id = -1
    self.inner_lrs = utils.Learner(
        (self.config.lagrangian_inner_lr, self.config.policy_inner_lr),
        next(self.rng_seq), self.config.inner_lr_opt,
        utils.get_mixed_precision_policy(config.precision))
    # Map task id to it's corresponding fine-tuned/adapted parameters.
    self.pi_posterior = [None] * self.config.task_batch_size

  def __call__(self, observation: np.ndarray, train: bool, adapt: bool, *args,
               **kwargs) -> np.ndarray:
    if train and self.buffer.full and self.query_buffer.full:
      assert adapt, (
          'Should train at the first step of adaptation (after filling up the '
          'buffer with adaptation and query data)')
      self.train(self.buffer.dump())
      self.logger.log_metrics(self.training_step)
    # Use the prior parameters on adaptation phase.
    policy_params = self.actor.params if adapt else self.task_posterior_params
    action = self.policy(observation, policy_params, next(self.rng_seq), train)
    return action

  def observe(self, transition: Transition, adapt: bool):
    if adapt:
      self.buffer.add(transition)
    else:
      self.query_buffer.add(transition)
    self.training_step += sum(transition.steps)

  def observe_task_id(self, task_id: Optional[str] = None):
    self.task_id = (self.task_id + 1) % self.config.task_batch_size
    self.buffer.set_task(self.task_id)
    self.query_buffer.set_task(self.task_id)

  def train(self, trajectory_data: TrajectoryData):
    support = trajectory_data
    query = self.query_buffer.dump()
    (self.lagrangian.state, self.actor.state, self.inner_lrs.state,
     info) = self.update_priors(self.lagrangian.state, self.actor.state,
                                self.inner_lrs.state, support, query)
    info['agent/lagrangian_lr'] = self.inner_lrs.params[0]
    info['agent/policy_lr'] = self.inner_lrs.params[0]
    for k, v in info.items():
      self.logger[k] = v.mean()

  def evaluate_support_and_query(
      self, support: TrajectoryData,
      query: TrajectoryData) -> [Evaluation, Evaluation]:
    support_eval = self.adapt_critics_and_evaluate(support.o, support.r,
                                                   support.c)
    query_eval = self.adapt_critics_and_evaluate(query.o, query.r, query.c)
    return support_eval, query_eval

  @functools.partial(jax.jit, static_argnums=0)
  def update_priors(self, lagrangian_state: LearningState,
                    actor_state: LearningState, inner_lr_state: LearningState,
                    support: TrajectoryData,
                    query: TrajectoryData) -> [LearningState, dict]:
    support_eval, query_eval = self.evaluate_support_and_query(support, query)
    old_pi = self.actor.apply(actor_state.params, support.o[:, :, :-1])
    old_pi_logprob = old_pi.log_prob(support.a)
    old_pi_query = self.actor.apply(actor_state.params, query.o[:, :, :-1])
    old_pi_query_logprob = old_pi_query.log_prob(query.a)

    def cond(val):
      iter_, *_, info = val
      kl = info['agent/actor/delta_kl']
      # Returns Truthy if iter is smaller than pi_iters and kl smaller than
      # kl threshold to continue iterating
      return jax.lax.bitwise_not(
          jax.lax.bitwise_or(kl > self.config.kl_margin * self.config.target_kl,
                             iter_ == self.config.pi_iters))

    def body(val):
      (iter_, lagrangian_state, actor_state, lr_state, _) = val
      loss, grads = jax.value_and_grad(self.meta_loss, (0, 1, 2))(
          lagrangian_state.params, actor_state.params, lr_state.params, support,
          query, support_eval, query_eval, old_pi_logprob, old_pi_query_logprob)
      lagrangian_grads, pi_grads, lr_grads = grads
      new_actor_state = self.actor.grad_step(pi_grads, actor_state)
      new_lagrangian_state = self.lagrangian.grad_step(lagrangian_grads,
                                                       lagrangian_state)
      new_lr_state = self.inner_lrs.grad_step(lr_grads, inner_lr_state)
      pi = self.actor.apply(actor_state.params, support.o[:, :, :-1])
      kl_d = old_pi.kl_divergence(pi).mean()
      report = {
          'agent/actor/loss': loss,
          'agent/actor/grad': optax.global_norm(grads),
          'agent/actor/entropy': pi.entropy().mean(),
          'agent/actor/delta_kl': kl_d
      }
      out = (iter_ + 1, new_lagrangian_state, new_actor_state, new_lr_state,
             report)
      return out

    init_state = (0, lagrangian_state, actor_state, inner_lr_state, {
        'agent/actor/loss': 0.,
        'agent/actor/grad': 0.,
        'agent/actor/entropy': 0.,
        'agent/actor/delta_kl': 0.
    })
    (iters, new_lagrangian_state, new_actor_state, new_lr_state,
     info) = jax.lax.while_loop(cond, body, init_state)  # noqa
    info['agent/actor/update_iters'] = iters
    new_lagrangian = self.lagrangian.apply(new_lagrangian_state.params)
    info['agent/lagrangian'] = new_lagrangian
    return new_lagrangian_state, new_actor_state, new_lr_state, info

  def meta_loss(self, lagrangian_prior: hk.Params, policy_prior: hk.Params,
                inner_lrs: Tuple[float, float], support: TrajectoryData,
                query: TrajectoryData, support_eval: Evaluation,
                query_eval: Evaluation, old_pi_logprob: jnp.ndarray,
                old_pi_query_logprob: jnp.ndarray):
    lagrangian_lr, pi_lr = inner_lrs
    constraint = support.c.sum(2).mean(1)
    lagrangian_posterior, pi_posteriors = self.adaptation_step(
        lagrangian_prior, policy_prior, lagrangian_lr, pi_lr,
        support.o[:, :, :-1], support.a, support_eval.advantage,
        support_eval.cost_advantage, constraint, old_pi_logprob)
    # vmap lagrangian and policy loss over the task axis.
    loss = jax.vmap(self.policy_loss)
    lagrangian = jax.vmap(self.lagrangian.apply)
    lagrangian_posterior = jnn.softplus(lagrangian(lagrangian_posterior))
    return loss(pi_posteriors, query.o[:, :, :-1], query.a,
                query_eval.advantage, query_eval.cost_advantage,
                lagrangian_posterior, old_pi_query_logprob).mean()

  def adapt(self, observation: np.ndarray, action: np.ndarray,
            reward: np.ndarray, cost: np.ndarray):
    constraint = cost.sum(2).mean(1)
    (advantage, return_, cost_advantage,
     cost_return) = self.adapt_critics_and_evaluate(observation, reward, cost)
    lagrangian_lr, pi_lr = self.inner_lrs.params
    old_pi = self.actor.apply(self.actor.params, observation[:, :, :-1])
    old_pi_logprob = old_pi.log_prob(action)
    _, pi_posteriors = self.adaptation_step(self.lagrangian.params,
                                            self.actor.params, lagrangian_lr,
                                            pi_lr, observation[:, :, :-1],
                                            action, advantage, cost_advantage,
                                            constraint, old_pi_logprob)
    for i in range(self.config.task_batch_size):
      self.pi_posterior[i] = jax.tree_util.tree_map(lambda leaf: leaf[i],
                                                    pi_posteriors)

  @functools.partial(jax.jit, static_argnums=0)
  def adaptation_step(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, lagrangian_lr: float,
                      policy_lr: float, observation: np.ndarray,
                      action: np.ndarray, advantage: np.ndarray,
                      cost_advantage: np.ndarray, constraint: np.ndarray,
                      old_pi_log_prob: np.ndarray) -> [hk.Params, hk.Params]:
    per_task_adaptation = jax.vmap(
        functools.partial(self.task_adaptation, lagrangian_prior, policy_prior,
                          lagrangian_lr, policy_lr))
    return per_task_adaptation(observation, action, advantage, cost_advantage,
                               constraint, old_pi_log_prob)

  def task_adaptation(self, lagrangian_prior: hk.Params,
                      policy_prior: hk.Params, lagrangian_lr: float,
                      pi_lr: float, observation: jnp.ndarray,
                      action: jnp.ndarray, advantage: jnp.ndarray,
                      cost_advantage: jnp.ndarray, constraint: jnp.ndarray,
                      old_pi_log_prob: np.ndarray) -> [hk.Params, hk.Params]:
    """
    Finds policy's and lagrangian MAP paramters for a single task.
    """

    def grad_step(prior: Tuple[hk.Params, hk.Params], _):
      lagrangian_prior, policy_prior = prior
      if self.safe:

        def lagrangian_loss(params):
          return -(self.lagrangian.apply(params) *
                   (constraint - self.config.cost_limit))[0]

        lagrangian_grads = jax.grad(lagrangian_loss)(lagrangian_prior)
        lagrangian_posterior = utils.gradient_descent(lagrangian_grads,
                                                      lagrangian_prior,
                                                      lagrangian_lr)
        lagrangian = jnn.softplus(self.lagrangian.apply(lagrangian_posterior))
      else:
        lagrangian = 0.
        lagrangian_posterior = lagrangian_prior
      policy_grads = jax.grad(self.policy_loss)(policy_prior, observation,
                                                action, advantage,
                                                cost_advantage, lagrangian,
                                                old_pi_log_prob)
      policy_posterior = utils.gradient_descent(policy_grads, policy_prior,
                                                pi_lr)
      return (lagrangian_posterior, policy_posterior), None

    (lagrangian_posteriors,
     policy_posteriors), _ = jax.lax.scan(grad_step,
                                          (lagrangian_prior, policy_prior),
                                          jnp.arange(self.config.inner_steps))
    return lagrangian_posteriors, policy_posteriors

  def adapt_critics_and_evaluate(self, observation: np.ndarray,
                                 reward: np.ndarray,
                                 cost: np.ndarray) -> Evaluation:
    # Find parameters that fit well the critics for the new tasks.
    return_, cost_return = self.returns(reward, cost)
    critic_states, _ = self.update_critic(self.critic.state,
                                          observation[:, :, :-1], return_)
    if self.safe:
      safety_critic_states, _ = self.update_safety_critic(
          self.safety_critic.state, observation[:, :, :-1], cost_return)
    else:
      safety_critic_states = critic_states
    # Evaluate with the task specific value functions.
    return self.evaluate_with_safety(critic_states.params,
                                     safety_critic_states.params, observation,
                                     reward, cost)

  @functools.partial(jax.jit, static_argnums=0)
  def evaluate_with_safety(self, critic_params: hk.Params,
                           safety_critic_params: hk.Params,
                           observation: jnp.ndarray, reward: jnp.ndarray,
                           cost: jnp.ndarray) -> Evaluation:
    task_batched = jax.vmap(super(MamlPpoLagrangian, self).evaluate_with_safety)
    return task_batched(critic_params, safety_critic_params, observation,
                        reward, cost)

  @functools.partial(jax.jit, static_argnums=0)
  def update_critic(self, state: LearningState, observation: jnp.ndarray,
                    return_: jnp.ndarray) -> [LearningState, dict]:
    partial = functools.partial(
        super(MamlPpoLagrangian, self).update_critic, state)
    # vmap over the task axis.
    return jax.vmap(partial)(observation, return_)

  @functools.partial(jax.jit, static_argnums=0)
  def update_safety_critic(self, state: LearningState, observation: jnp.ndarray,
                           cost_return: jnp.ndarray) -> [LearningState, dict]:
    partial = functools.partial(
        super(MamlPpoLagrangian, self).update_safety_critic, state)
    # vmap over the task axis.
    return jax.vmap(partial)(observation, cost_return)

  @functools.partial(jax.jit, static_argnums=0)
  def returns(self, rewards: jnp.ndarray, costs: jnp.ndarray):
    reward_return = discounted_cumsum(rewards, self.config.discount)
    cost_return = discounted_cumsum(costs, self.config.cost_discount)
    return reward_return, cost_return

  @property
  def task_posterior_params(self):
    if None in self.pi_posterior:
      return self.actor.params
    return self.pi_posterior[self.task_id]
