#!/bin/bash
# sh testTD3ucb 0 0.01 5
python main.py --env Ant-v3 --policy TD3ucb --seed $1 --dropout-rate $2 --ucb-times $3
python main.py --env HalfCheetah-v3 --policy TD3ucb --seed $1 --dropout-rate $2 --ucb-times $3
python main.py --env Walker2d-v3 --policy TD3ucb --seed $1 --dropout-rate $2 --ucb-times $3
python main.py --env Hopper-v3 --policy TD3ucb --seed $1 --dropout-rate $2 --ucb-times $3