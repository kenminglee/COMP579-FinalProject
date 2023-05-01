import gymnasium as gym
import numpy as np


def make_discrete_env(env_id, seed, idx, capture_video, run_name, normalize_obs=False, **kwargs):
    def thunk():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_continuous_env(env_id, seed, idx, capture_video, run_name, gamma, **kwargs):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array", **kwargs)
        else:
            env = gym.make(env_id, **kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# def vec_env_loop(num_steps:int, prev_obs, agent: BaseAgent, envs: gym.vector.SyncVectorEnv, train:bool=True):
#     obs = prev_obs
#     a = agent.choose_action(obs, train=train)  
#     for _ in range(num_steps):
#         next_obs, r, terminated, truncated, info = envs.step(a)
#         a = agent.choose_action(next_obs, train=train) 
#         if train:
#             agent.learn(obs, a, r, next_obs, terminated, truncated, info)
#         obs = next_obs


# envs = gym.vector.SyncVectorEnv(
#         [make_env("CartPole-v1", 0 + i, i, False, None) for i in range(2)]
#     )
# print(envs.reset())
# print(envs.single_observation_space)
# print(envs.step(envs.action_space.sample()))