#!/bin/sh

module add anaconda/2021.05
module add cuda/11.3

source activate metatsp 
python ../main.py --problem_size 200
