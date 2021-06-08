#!/bin/bash
python evaluation.py --logdir logs --id <insert exp id> --configs defaults moving_mnist --use_wandb False --steps 1e5 --batch_size 50