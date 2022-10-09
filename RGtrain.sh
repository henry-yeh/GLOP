#!/bin/sh

module add anaconda/2021.05
module add cuda/11.3

source activate metatsp 
python ./run.py --graph_size 100 --lr_decay 0.99 --run_name RG-train-100-6layer --load_path pretrained_LCP/constructions/Reviser-6-scale/reviser_100/epoch-400.pt --n_epochs 300 --checkpoint_epochs 50 --no_tensorboard --no_progress_bar --n_encode_layers 6

