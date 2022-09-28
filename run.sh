#!/bin/sh

module add anaconda/2021.05
module add cuda/11.3

source activate metatsp 
python ./run.py --data_distribution scale --graph_size 20 --run_name 3-scale-NMT-20

