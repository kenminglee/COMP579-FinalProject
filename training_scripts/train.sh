#!/bin/bash

# python3 training_scripts/train.py --seed 1 --env-id StoreEnv-v1 --total-timesteps 2000000 --num-envs 4 --num-steps 128 --num-minibatches 1 --gae-lambda 0.95 --gamma 0.99 --update-epochs 20 --ent-coef 0.05 --learning-rate 1e-3 --anneal-lr False --clip-coef 0.2 --video --cuda False  --env-kwargs obs_type=state irrational_weight=0.0 layout=default #--track

for lambda in {0.0,0.3,0.5,0.7,1.0}
do 
python3 training_scripts/train_tabular.py --seed 1 --env-id StoreEnv-v1 --total-timesteps 2000000 --learning-rate 1.0 --gamma 0.99 --epsilon 0.1 --init-val 0.0 --video --env-kwargs obs_type=tabular irrational_weight=$lambda layout=interpolate --track

python3 training_scripts/train_tabular.py --seed 1 --env-id StoreEnv-v1 --total-timesteps 2000000 --learning-rate 1.0 --gamma 0.99 --epsilon 0.1 --video --env-kwargs obs_type=tabular irrational_weight=$lambda layout=prefer-yellow --track
done
