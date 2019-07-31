#!/usr/bin/env bash
set -ex
python train.py --dataroot ~/data/celeb/ \
                --name gan_test \
                --model gan \
                --dataset_mode single \
                --load_size 128 \
                --crop_size 64 \
                --batch_size 32