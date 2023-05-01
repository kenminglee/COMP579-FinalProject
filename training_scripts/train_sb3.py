import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import TensorBoardOutputFormat

from store_env.environment.wrapper import make_discrete_env, make_continuous_env

from stable_baselines3.common.callbacks import BaseCallback

def parse_args_sb3_ppo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, 
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="COMP579-Final-Project",
        help="the wandb's project name")
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=300000,
        help="total timesteps of the experiments")

    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-range-vf", type=float, default=None, 
        help="Clip range for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    return args

class TbRewardLogger(BaseCallback):

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.episode_count = 0
        # Get access to the low-level tensorboard summary writer
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter: TensorBoardOutputFormat = next(formatter for formatter in self.logger.output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if any(self.locals["dones"]): # If there are episode(s) that ended
            episodic_returns, episode_lengths = [], []
            for info in self.locals["infos"]:
                if 'episode' in info:
                    episodic_returns.append(info['episode']['r'])
                    episode_lengths.append(info['episode']['l'])
            self.episode_count += len(episodic_returns)
            mean_rew, mean_length = np.mean(episodic_returns), np.mean(episode_lengths).astype(int)
            # print(f"{self.num_timesteps=}, {mean_rew=:.3}")
            self.tb_formatter.writer.add_scalar("charts/episodic_return", mean_rew, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("charts/episodic_length", mean_length, self.num_timesteps)
            self.tb_formatter.writer.add_scalar("charts/episode", self.episode_count, self.num_timesteps)
        
        return True


if __name__=="__main__":
    args = parse_args_sb3_ppo()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    assert args.env_id in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "HalfCheetah-v4", "MountainCarContinuous-v0"]

    if args.env_id in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1"]:
        envs = DummyVecEnv([make_discrete_env(args.env_id, args.seed + i, i, False, None, normalize_obs=False) for i in range(args.num_envs)])
    else:
        envs = DummyVecEnv([make_continuous_env(args.env_id, args.seed + i, i, False, None, args.gamma) for i in range(args.num_envs)])
   
    run_name = f'sb3ppo_{args.env_id}_{args.seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            tags=[args.env_id],
            group="SB3 PPO",
            name=run_name,
            save_code=False
        )
    if args.anneal_lr:
        lr_schedule = lambda frac: frac*args.learning_rate
    else:
        lr_schedule = lambda _: args.learning_rate


    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))
    batch_size = int((args.num_envs*args.num_steps)//args.num_minibatches)
    model = PPO("MlpPolicy", envs, verbose=1, 
                learning_rate=lr_schedule, 
                n_steps=args.num_steps,
                batch_size=batch_size,
                n_epochs=args.update_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_coef,
                clip_range_vf=args.clip_range_vf,
                normalize_advantage=args.norm_adv,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                tensorboard_log=f"runs/{run_name}",
                seed=args.seed,
                device=device,
                policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.total_timesteps, log_interval=1, callback=[TbRewardLogger()])