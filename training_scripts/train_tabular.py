from typing import Union, List
import time
import random

import numpy as np
import wandb

from store_env.algorithms.base_class import BaseAgent
from store_env.algorithms.qlearning import QLearning, parse_args_qlearning 
from store_env.environment.env import StoreEnv
from store_env.environment.wrapper import make_discrete_env

if __name__=="__main__":
    args = parse_args_qlearning()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_name = f'qlearning_{args.env_id}_{args.seed}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'

    assert args.env_id in ["StoreEnv-v1"]
    if args.env_id=="StoreEnv-v1" and 'obs_type' not in args.env_kwargs:
        args.env_kwargs['obs_type'] = 'tabular'
    env = make_discrete_env(args.env_id, args.seed, 0, args.video, run_name, **args.env_kwargs)()

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            tags=[args.env_id],
            group="Q-learning",
            name=run_name,
            save_code=False
        )

    agent = QLearning(env.action_space.n,learning_rate=args.learning_rate, discount=args.gamma,epsilon=args.epsilon, seed=args.seed, init_val=args.init_val, run_name=f'runs/{run_name}')
    s, info = env.reset()
    s = tuple(s)

    cum_rew = 0
    ep = 0
    for step in range(args.total_timesteps):
        a = agent.choose_action(s)  
        s_, r, terminated, truncated, info = env.step(a)
        s_ = tuple(s_)
        agent.learn(s, a, r, s_, terminated, truncated, info)
        s = s_
        cum_rew += r
        if terminated or truncated: 
            ep += 1
            if ep%10==0:
                print(f"{ep=}, {step=}, {cum_rew=}")
            s, info = env.reset()
            s = tuple(s)
            cum_rew = 0
    
    if args.video and args.track and ep>=1000:
        wandb.log({"video": wandb.Video(f"videos/{run_name}/rl-video-episode-{(ep//1000)*1000}.mp4")})
    if args.track:
        wandb.finish()