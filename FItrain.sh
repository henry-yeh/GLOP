#!/bin/sh

module add anaconda/2021.05
module add cuda/11.3

source activate metatsp 
python ./run.py --FI_train --graph_size 50 --lr_decay 0.99 --FI_path data/FI_train_tsp/500_FI50_seed1235.pkl --run_name FI-train-50-6layer --load_path pretrained_LCP/constructions/Reviser-6-scale/reviser_50/epoch-400.pt --n_epochs 300 --checkpoint_epochs 50 --no_tensorboard --no_progress_bar --n_encode_layers 6

