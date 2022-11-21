#!/bin/bash
POLICY=${1?"ERROR! sh test.sh [POLICY] [VER] [RANDOM SEED]"}
VER=${2?"ERROR! sh test.sh [POLICY] [VER] [RANDOM SEED]"}
SEED=${3?"ERROR! sh test.sh [POLICY] [VER] [RANDOM SEED]"}

python main.py --env Ant-v3 --policy $1 --seed $3 --qweight 0.25 --ver $2 --actor-num $4 --critic-num $5
python main.py --env HalfCheetah-v3 --policy $1 --seed $3 --qweight 0.1 --ver $2 --actor-num $4 --critic-num $5
python main.py --env Walker2d-v3 --policy $1 --seed $3 --qweight 0.1 --ver $2 --actor-num $4 --critic-num $5
python main.py --env Hopper-v3 --policy $1 --seed $3 --qweight 0.15 --ver $2 --actor-num $4 --critic-num $5
python main.py --env Humanoid-v3 --policy $1 --seed $3 --actor-lr 3e-4 --critic-lr 3e-4 --steps 3000000 --qweight 0.05 --ver $2 --actor-num $4 --critic-num $5