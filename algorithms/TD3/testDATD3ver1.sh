#!/bin/bash
# sh testDATD3ver1.sh [seed]
python main.py --env BipedalWalker-v3 --policy DATD3ver1 --seed $1
python main.py --env Ant-v3 --policy DATD3ver1 --seed $1
python main.py --env HalfCheetah-v3 --policy DATD3ver1 --seed $1
python main.py --env Walker2d-v3 --policy DATD3ver1 --seed $1
python main.py --env Hopper-v3 --policy DATD3ver1 --seed $1