import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import wandb

from store_env.algorithms.ppo import PPO, PPOContinuousNetwork, PPODiscreteNetwork, parse_args_ppo
from store_env.environment.wrapper import make_discrete_env, make_continuous_env


if __name__=="__main__":
    args = parse_args_ppo()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    assert args.env_id in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "HalfCheetah-v4", "MountainCarContinuous-v0", "StoreEnv-v1"]

    run_name = f'customppo_{args.env_id}_{args.seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'

    if args.env_id=="StoreEnv-v1" and 'obs_type' not in args.env_kwargs:
        args.env_kwargs['obs_type'] = 'state'
    if args.env_id in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "StoreEnv-v1"]:
        envs = gym.vector.SyncVectorEnv(
                [make_discrete_env(args.env_id, args.seed + i, i, args.video, run_name, normalize_obs=False, **args.env_kwargs) for i in range(args.num_envs)]
            )
    else:
        envs = gym.vector.SyncVectorEnv(
                [make_continuous_env(args.env_id, args.seed + i, i, args.video, run_name, args.gamma, **args.env_kwargs) for i in range(args.num_envs)]
            )
    envs.reset(seed=args.seed)
    
    env = gym.make(args.env_id)
    network = PPODiscreteNetwork if isinstance(env.action_space, gym.spaces.Discrete) else PPOContinuousNetwork

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            tags=[args.env_id],
            group="PPO (ours)",
            name=run_name,
            save_code=False
        )

    agent = PPO(
        single_obs_space=envs.single_observation_space,
        single_action_space=envs.single_action_space,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_env_steps=args.total_timesteps,
        num_minibatches=args.num_minibatches,
        lr=args.learning_rate,
        anneal_lr=args.anneal_lr,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        device=device,
        network=network,
        seed=args.seed,
        torch_cuda_deterministic=args.torch_deterministic,
        run_name=f'runs/{run_name}'
    )

    # Env loop
    conv_to_tensor = lambda x: torch.tensor(x).to(device)
    s, info = envs.reset()
    s = conv_to_tensor(s)
    for _ in range(int(args.total_timesteps//args.num_envs)):
        a = agent.choose_action(s)
        s_, r, terminated, truncated, info = envs.step(a.cpu().numpy())
        s_ = conv_to_tensor(s_)
        agent.learn(s, a, conv_to_tensor(r).view(-1), s_, conv_to_tensor(terminated), conv_to_tensor(truncated), info)
        s = s_