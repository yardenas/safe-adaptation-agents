import os

from gym.wrappers import TimeLimit

from safe_adaptation_agents import agents
from safe_adaptation_agents import config as options
from safe_adaptation_agents.trainer import Trainer


def make_env(config):
  import safe_adaptation_gym
  env = safe_adaptation_gym.make(config.robot, config.task)
  env = TimeLimit(env, config.time_limit)
  return env


def main():
  config = options.load_config([
      '--configs', 'defaults', 'no_adaptation', '--agent', 'ppo_lagrangian',
      '--num_trajectories', '30', '--vf_iters', '80', '--pi_iters', '80',
      '--eval_every', '5', '--eval_trials', '0',
      '--test_driver.adaptation_steps', '3000', 'train_driver.adaptation_steps',
      '45000', '--actor_opt.lr', '0.02', '--critic_opt.lr', '0.02',
      '--discount', '0.95', '--epochs', '50'
  ])
  if not config.jit:
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', True)
  path = os.path.join(config.log_dir, 'state.pkl')
  with Trainer.from_pickle(config) if os.path.exists(path) else Trainer(
      config=config, make_agent=agents.make,
      make_env=lambda: make_env(config)) as trainer:
    objective, cost = trainer.train()
  assert objective > 185.
  assert cost == 0.


if __name__ == '__main__':
  main()
