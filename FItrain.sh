#!/bin/sh

module add anaconda/2021.05
module add cuda/11.3

source activate metatsp 
python ./run.py --FI_train --graph_size 20 --lr_decay 0.9 --run_name FI-train-20 --load_path pretrained_LCP/Reviser-3-scale-NMT/reviser_20/epoch-199.pt --n_epochs 20 --no_tensorboard --no_progress_bar

