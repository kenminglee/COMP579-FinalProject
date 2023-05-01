# StoreEnv
A fast and customizable gymnasium-compatible retail store environment for reinforcement learning research
# Results
- All plots from the paper are available [here](https://wandb.ai/kenminglee/COMP579-Final-Project/reports/Results-Plots--Vmlldzo0MTY3MjUw#benchmarking-ppo)
- Video replays of our agent on the custom `StoreEnv-v1` environment are available [here](https://wandb.ai/kenminglee/COMP579-Final-Project/reports/Videos--Vmlldzo0MTc4MTg3) 
# User Guide
## Installation
1. Install [Poetry](https://python-poetry.org/docs/)
2. Do `poetry install --all-extras` to install the all dependencies and the `store_env` Python library. Alternatively, do `poetry install` instead if we do not need video recordings and do not plan to use Stable Baselines3 for benchmarking purposes.
## Reproducing our results
### Benchmarking PPO implementation
To reproduce our PPO benchmarking results, run the `benchmark.sh` script as follows:
```sh
./training_scripts/benchmark.sh
```
### Results on StoreEnv
To reproduce results of our two sets of experiments on the `StoreEnv-v1` environment, run the `train.sh` script as follows:
```sh
./training_scripts/train.sh
```

## Using StoreEnv 
Below is the minimal example to use StoreEnv as a standalone Gymnasium environment:
```python
import gymnasium
import store_env

env = gymnasium.make('StoreEnv-v1')
env.reset()
for _ in range(1000):
    _, r, terminated, truncated, _ = env.step(action=env.action_space.sample())
    if terminated or truncated:
        env.reset()
``` 
If we want to render the environment and control with our keyboard, please run the `store_env/environment/manual_control.py` Python script.


# Credits
## Environment
- [Minigrid](https://github.com/Farama-Foundation/Minigrid) 
## Algorithm Implementation
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
