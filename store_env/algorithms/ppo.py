import argparse
import os
import random
import time
from distutils.util import strtobool
import warnings
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from store_env.algorithms.base_class import BaseAgent
from store_env.environment.utils import add_env_kwargs_parser
    

def parse_args_ppo():
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
    parser = add_env_kwargs_parser(parser)
    parser.add_argument("--total-timesteps", type=int, default=300000,
        help="total timesteps of the experiments")
    parser.add_argument("--video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if flag exists, video recording will be enabled")

    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOBaseNetwork(nn.Module):
    
    def get_value(self, x):
        ''' returns value of state (V(x))
        '''
        raise NotImplementedError

    def get_action_and_value(self, x, action=None):
        ''' returns action (pi(x)), logprob (log pi(x)), entropy (H(pi(x))), value of state (V(x))
        '''
        raise NotImplementedError
    
class PPOContinuousNetwork(PPOBaseNetwork):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(act_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(act_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float()
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class PPODiscreteNetwork(PPOBaseNetwork):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

class PPO(BaseAgent):
    def __init__(self, 
        single_obs_space: gym.Space, 
        single_action_space: gym.Space, 
        num_envs:int,
        num_steps: int,
        total_env_steps: int, # (assume env not vectorized) only matters if anneal_lr=True
        num_minibatches: int, # 1 means full batch, batch_size = num_envs * num_steps
        lr: float,
        anneal_lr: bool,
        gae_lambda: float, # -1 means disable gae
        gamma: float,
        update_epochs: int, 
        norm_adv: bool,
        clip_coef: float,
        clip_vloss: bool,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float, # -1 to disable grad norm clipping
        target_kl: float,
        device: torch.device,
        network: Optional[nn.Module] = None, # Set automatically to default if none
        seed:int=1,
        torch_cuda_deterministic:bool = True,
        run_name: str = None # if none we'll create our own
        ):
        ''' Note: we are using internal counter to count steps for updates. If not learning (only exploitation), use choose_action() with train=False and do not call learn()
        '''
        self.seed = seed
        self.torch_cuda_deterministic = torch_cuda_deterministic
        self._seed()
        self.obs_space = single_obs_space
        self.action_space = single_action_space
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        self.batch_size: int = int(num_envs*num_steps)
        if self.batch_size%num_minibatches!=0:
            warnings.warn("Batch size is not divisible by num_minibatches", UserWarning)
        self.minibatch_size: int = int(self.batch_size // num_minibatches)

        if network is not None:
            self.agent = network(obs_space=self.obs_space, act_space=self.action_space).to(device)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.agent = PPODiscreteNetwork(obs_space=self.obs_space, act_space=self.action_space).to(device)
        elif isinstance(self.action_space, gym.spaces.Box):
            self.agent = PPOContinuousNetwork(obs_space=self.obs_space, act_space=self.action_space).to(device)
        else:
            raise TypeError(f"Default networks only support Discrete and Box action space. For other action spaces, please use a custom network.")

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        self.total_num_updates = total_env_steps // self.batch_size
        self.step = 0
        self.global_step=0
        self.update_count = 0
        self.episode_count = 0
        
        self.writer = SummaryWriter(run_name if run_name is not None else f'runs/{time.strftime("%d-%m-%Y_%H-%M-%S")}')
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.__dict__.items()])),
        )

        self.init_buffer()

    def _seed(self):
        random.seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_cuda_deterministic

    def init_buffer(self):
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space.shape).to(self.device)
        self.next_obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.next_obs_values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def choose_action(self, s, train:bool=True): 
        ''' if train=True, save values and logprobs
            if train=False, return greedy action (and discard values and logprobs)
        '''
        assert train # for now assert train==True
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(s)
        self.next_obs_values[self.step-1] = value.flatten()
        self.logprobs[self.step] = logprob
        return action

    def learn(self, s, a, r, s_, terminated, truncated, info): # Implicitly count step number here!
        self.obs[self.step] = s
        self.actions[self.step] = a
        self.rewards[self.step] = r
        self.next_obs[self.step] = s_
        self.dones[self.step] = torch.logical_or(terminated, truncated)

        self.step += 1
        self.global_step += self.num_envs
        
        episodic_returns, episode_lengths = [], []
        for item in info.get('final_info', []):
            if item and "episode" in item.keys():
                episodic_returns.append(item['episode']['r'])
                episode_lengths.append(item['episode']['l'])

        # Equivalent to the following one-liner (tho less readable)
        # episodic_returns, episode_lengths = zip(*[(item['episode']['r'], item["episode"]["l"]) for item in info.get('final_info', []) if item and "episode" in item.keys()])

        if episodic_returns:
            # Take the mean over number of environments (when multiple envs are done at the ame step)
            mean_rew, mean_length = np.mean(episodic_returns), np.mean(episode_lengths).astype(int)
            self.episode_count += len(episodic_returns)
            print(f"global_step={self.global_step}, {mean_rew=}")
            self.writer.add_scalar("charts/episodic_return", mean_rew, self.global_step)
            self.writer.add_scalar("charts/episodic_length", mean_length, self.global_step)
            self.writer.add_scalar("charts/episode", self.episode_count, self.global_step)
        
        if self.step >= self.num_steps:
            self.update(self.global_step)
            self.step=0

    def update(self, global_step):
        with torch.no_grad():
            # v(s1),v(s2),...,v(st),v(s0) -> v(s0),v(s1),...,v(s{t-1}),v(st)
            values = self.next_obs_values.roll(1, dims=0)
            nextvalues = self.next_obs_values
            nextvalues[-1] = self.agent.get_value(self.next_obs[self.step-1]).reshape(1, -1)
            nextnonterminal = 1.0 - self.dones
            delta = self.rewards + self.gamma * nextvalues * nextnonterminal - values
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                advantages[t] = lastgaelam = delta[t] + self.gamma * self.gae_lambda * nextnonterminal[t] * lastgaelam
            returns = advantages + values # Recall: A = G - V (where G is discounted cum sum)
        
        # flatten the batch (since our environment is vectorized)
        b_obs = self.obs.reshape((-1,) + self.obs_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizaing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            self.rng.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm!=-1:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.anneal_lr:
            self.update_count += 1
            frac = 1.0 - (self.update_count - 1.0) / self.total_num_updates
            lrnow = frac * self.lr
            self.optimizer.param_groups[0]["lr"] = lrnow
        
        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("train/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("train/policy_gradient_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("train/entropy_loss", entropy_loss.item(), global_step)
        self.writer.add_scalar("train/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("train/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("train/clip_fraction", np.mean(clipfracs), global_step)
        self.writer.add_scalar("train/explained_variance", explained_var, global_step)