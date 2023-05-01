#!/bin/bash

for FILENAME in "train.py" "train_sb3.py"
do 
python3 training_scripts/$FILENAME --seed $1 --env-id CartPole-v1 --total-timesteps 100000 --num-envs 8 --num-steps 32 --num-minibatches 1 --gae-lambda 0.8 --gamma 0.98 --update-epochs 20 --ent-coef 0.0 --learning-rate 1e-3 --anneal-lr True --clip-coef 0.2 --track

python3 training_scripts/$FILENAME --seed $1 --env-id Acrobot-v1 --total-timesteps 1000000 --num-envs 16 --num-steps 256 --num-minibatches 64 --gae-lambda 0.94 --gamma 0.99 --update-epochs 4 --ent-coef 0.0 --track

python3 training_scripts/$FILENAME --seed $1 --env-id HalfCheetah-v4 --total-timesteps 1000000 --num-envs 1 --num-steps 2048 --num-minibatches 32 --update-epochs 10 --learning-rate 3e-4 --track 
done

# Diff hyperparameters required for MountainCarContinuous-v0
python3 training_scripts/train.py --seed $1 --env-id MountainCarContinuous-v0 --total-timesteps 40000 --num-envs 32 --num-steps 8 --num-minibatches 1 --gae-lambda 0.9 --gamma 0.9999 --update-epochs 10 --ent-coef 0.00429 --learning-rate 7.77e-05 --anneal-lr False --clip-coef 0.1 --max-grad-norm 5 --vf-coef 0.19 --track

python3 training_scripts/train_sb3.py --env-id MountainCarContinuous-v0 --total-timesteps 40000 --num-envs 1 --num-steps 8 --num-minibatches 1 --gae-lambda 0.9 --gamma 0.9999 --update-epochs 10 --ent-coef 0.00429 --learning-rate 7.77e-05 --anneal-lr False --clip-coef 0.1 --max-grad-norm 5 --vf-coef 0.19 --track --seed $1 
