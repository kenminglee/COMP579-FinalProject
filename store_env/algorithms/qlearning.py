from collections import defaultdict
import random
import time
import argparse
from distutils.util import strtobool

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from store_env.algorithms.base_class import BaseAgent
from store_env.environment.utils import add_env_kwargs_parser

def parse_args_qlearning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, 
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="COMP579-Final-Project",
        help="the wandb's project name")
    parser.add_argument("--env-id", type=str, default="StoreEnv-v1",
        help="the id of the environment")
    parser = add_env_kwargs_parser(parser)
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if flag exists, video recording will be enabled")

    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=1,
        help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--epsilon", type=float, default=0.1,
        help="exploration rate")
    parser.add_argument("--init-val", type=float, default=0.0,
        help="initial value of qtable")
    args = parser.parse_args()
    return args

class QLearning(BaseAgent):
    def __init__(self, num_actions, learning_rate=0.1, epsilon=0.1,  discount=0.9, init_val=0., seed=1, run_name=None) -> None:
        self.init_val = init_val
        self.num_actions = num_actions
        self.alpha = np.float32(learning_rate)
        self.epsilon = np.float32(epsilon)
        self.gamma = np.float32(discount)
        self.reset(seed)
        
        self.writer = SummaryWriter(run_name if run_name is not None else f'runs/{time.strftime("%d-%m-%Y_%H-%M-%S")}')
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.__dict__.items()])),
        )

    def choose_action(self, s, train=True):
        q = self.q[s]
        if not train or random.random()>=self.epsilon:
            a = self.rng.choice(np.where(q==np.max(q))[0])
        else:
            a = self.rng.choice(range(self.num_actions))
        return a

    def learn(self, s, a, r, s_, terminated, truncated, info):
        done = terminated or truncated
        self.q[s][a] = self.q[s][a] + self.alpha*(r+(1-done)*self.gamma*self.q[s_].max()-self.q[s][a])
        self.global_step += 1
        if done:
            self.episode_count += 1
            self.writer.add_scalar("charts/episodic_return", info['episode']['r'], self.global_step)
            self.writer.add_scalar("charts/episodic_length", info['episode']['l'], self.global_step)
            self.writer.add_scalar("charts/episode", self.episode_count, self.global_step)
    
    def reset(self, seed):
        self.episode_count = 0
        self.global_step = 0
        self.q = defaultdict(lambda: np.full(self.num_actions, self.init_val, dtype=np.float32))
        self.rng = np.random.default_rng(seed=seed)
        random.seed(seed)

    