#!/bin/bash

python main.py --env Ant-v3 --policy TD3ver1 --seed $1
python main.py --env HalfCheetah-v3 --policy TD3ver1 --seed $1
python main.py --env Walker2d-v3 --policy TD3ver1 --seed $1
python main.py --env Hopper-v3 --policy TD3ver1 --seed $1