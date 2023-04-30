# Learning Global Partition and Local Construction for Solving Large-scale Routing Problems



## Dependencies

- Python>=3.8
- NumPy==1.23
- CUDA 11.7
- PyTorch 1.13.0
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.0.4
- SciPy
- tqdm
- [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
- Matplotlib


## How to Use

### Resources

- Download checkpoints from [checkpoints-downloading-link](https://drive.google.com/file/d/1u9-GVTMRux3rWGcbipSqyTyBx_V8pm9G/view?usp=sharing) and place them in `./pretrained`.

- Download test datasets from [test-datasets-downloading-link](https://drive.google.com/file/d/1WuICJGKRsiTjVTq7_ivh29wWShv8BRBO/view?usp=sharing) and place them in `./data`.

### Evaluation 
Our datasets are mostly generated using the code of [*Attention, Learn to Solve Routing Problems!*](https://github.com/wouterkool/attention-learn-to-route). To evaluate our method on your own datasets, use `--path PATH_OF_YOUR_DATASET`.

#### For TSP
```bash
# For TSP500:
python main.py --problem_size 500 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 64 --val_size 128 --decode_strategy greedy

# For TSP1000:
python main.py --problem_size 1000 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 32 --val_size 128 --decode_strategy greedy

# For TSP10k:
python main.py --problem_size 10000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1 --eval_batch_size 16 --val_size 16 --decode_strategy greedy

# For TSP100k:
python main.py --problem_size 100000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1 --eval_batch_size 1 --val_size 1 --decode_strategy greedy

# To conduct cross-distribution evaluation, e.g.:
python main.py --problem_size 100 --revision_lens 100 50 20 10 --revision_iters 20 10 10 5 --width 140 --eval_batch_size 100 --val_size 10000 --decode_strategy sampling --path data/tsp/tsp_uniform100_10000.pkl --no_aug --no_prune

# To reproduce the results of 49 TSPLib instances:
python eval_tsplib.py --eval_batch_size 1 --val_size 49 --path data/tsp/tsplib49.pkl --width 128 --decode_strategy greedy --no_prune
```


To reduce the inference duration, try:
```bash
# set
--width 1
# add
--no_aug
# less revisions, e.g.,
--revision_iters 5 5 5
```

#### For CVRP

```bash
# For CVRP1K using LKH-3 as sub-solver: 
python eval_cvrp.py --cpus 12 --problem_size 1000

# For CVRP1K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 1000 --revision_lens 20 --revision_iters 5

# For CVRP2K using LKH-3 as sub-solver: 
python eval_cvrp.py --cpus 12 --problem_size 2000

# For CVRP2K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 2000 --revision_lens 50 20 --revision_iters 5 5

# For CVRP5K using LKH-3 as sub-solver
python eval_cvrp.py --cpus 12 --problem_size 5000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt

# For CVRP5K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 5000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt --revision_lens 20 --revision_iters 5

# For CVRP7K using LKH-3 as sub-solver
python eval_cvrp.py --cpus 12 --problem_size 7000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt

# For CVRP7K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 7000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt --revision_lens 20 --revision_iters 5
```


#### For PCTSP

```bash
# e.g., for PCTSP500
python main.py --problem_type pctsp --problem_size 500 --n_subset 10 --eval_batch_size 50 --val_size 100 --revision_iters 10 10 5 --revision_lens 100 50 20

# set n_subset = 1 for greedy mode
--n_subset 1
```

### Training

Please refer to `./local_construction/` and `./heatmap/`.

## Acknowledgements

* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/alstn12088/LCP
* https://github.com/Spider-scnu/TSP
* https://github.com/jieyibi/AMDKD
