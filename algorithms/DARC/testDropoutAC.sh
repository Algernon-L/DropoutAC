#!/bin/bash
python main.py --env Ant-v3 --policy DropoutAC --seed $1 --qweight 0.25 --ver $2
python main.py --env HalfCheetah-v3 --policy DropoutAC --seed $1 --qweight 0.1 --ver $2
python main.py --env Walker2d-v3 --policy DropoutAC --seed $1 --qweight 0.1 --ver $2
python main.py --env Hopper-v3 --policy DropoutAC --seed $1 --qweight 0.15 --ver $2
#python main.py --env Humanoid-v3 --policy DropoutAC --seed $1 --actor-lr 3e-4 --critic-lr 3e-4 --steps 3000000 --qweight 0.05 --ver $2