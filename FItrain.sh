#!/bin/sh

module add anaconda/2021.05
module add cuda/11.3

source activate metatsp 
python ./run.py --FI_train --graph_size 100 --lr_decay 0.99 --FI_path data/FI_train_tsp/500_FI100_seed1235.pt --run_name FI-train-100 --load_path pretrained_LCP/Reviser-scale/reviser_100/epoch-200.pt --n_epochs 300 --checkpoint_epochs 50 --no_tensorboard --no_progress_bar --n_encode_layers 6

