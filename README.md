# LoPo: Learning Local Policy to Solve Travelling Salesman Problem with an Arbitrary Scale and Distribution



## Dependencies

- Python>=3.8
- NumPy==1.23
- CUDA 11.0
- PyTorch 1.7.0
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.0.4
- SciPy
- tqdm
- [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
- Matplotlib


## How to Use

### Resources

- Download reviser checkpoints from [reviser-checkpoints-downloading-link](https://drive.google.com/file/d/1a5XthsAPr29bl-U5hD5rGk2Oqll2OqNq/view?usp=share_link) and place the checkpoints in `./pretrained/Reviser-stage2`.

- Download test datasets from [TSP-test-datasets-downloading-link](https://drive.google.com/file/d/1KfCuASdp5N1OH7m41zZ0j2D_zvLjhB2O/view?usp=share_link) and place the test datasets in `./data/tsp`.

- Download test datasets from [CVRP-test-datasets-downloading-link](https://drive.google.com/file/d/1yni24PcZJzOc8RKE4ZAhrPad7j7aZzJb/view?usp=sharing) and place the test datasets in `./data/vrp`.

- Download test datasets from [PCTSP-test-datasets-downloading-link](https://drive.google.com/file/d/1xkxDgr5xOoTm7bZEIWo3vOOIyoyx5naO/view?usp=sharing) and place the test datasets in `./data/pctsp`.


### Evaluation on TSP

```bash
# To reproduce the best result on 10k TSP20: 
python main.py --eval_batch_size 1000  --val_size 10000 --problem_size 20 --revision_lens 20 10 --revision_iters 10 5 --width 80 --decode_strategy sampling --no_aug

# To reproduce the best result on 10k TSP50: 
python main.py --eval_batch_size 500  --val_size 10000 --problem_size 50 --revision_lens 50 20 --revision_iters 30 10 --width 80 --decode_strategy sampling --no_aug

# To reproduce the best result on 10k TSP100: 
python main.py --problem_size 100 --revision_lens 100 50 20 10 --revision_iters 20 10 10 5 --width 80 --eval_batch_size 200 --val_size 10000 --decode_strategy sampling --no_aug

# To reproduce the best result on 128 TSP200: 
python main.py --problem_size 200 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 128 --val_size 128 --decode_strategy greedy

# To reproduce the best result on 128 TSP500:
python main.py --problem_size 500 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 64 --val_size 128 --decode_strategy greedy

# To reproduce the best result on 128 TSP1000:
python main.py --problem_size 1000 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 32 --val_size 128 --decode_strategy greedy

# To reproduce the best result on 16 TSP10k:
python main.py --problem_size 10000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 4 --val_size 16 --decode_strategy greedy

# To reproduce the best result on 1 TSP100k:
python main.py --problem_size 100000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1 --eval_batch_size 1 --val_size 1 --decode_strategy greedy

# To conduct cross-distribution evaluation, e.g.:
python main.py --problem_size 100 --revision_lens 100 50 20 10 --revision_iters 20 10 10 5 --width 140 --eval_batch_size 100 --val_size 10000 --decode_strategy sampling --path data/tsp/tsp_uniform100_10000.pkl  --no_aug

# To reproduce the results of 49 TSPlib instances:
python eval_tsplib.py --eval_batch_size 1 --val_size 49 --path data/tsp/tsplib49.pkl --width 128 --decode_strategy greedy
```


To reduce the inference duration, try:
```bash
# set
--width 1
# add
--no_aug
```

### Evaluation on CVRP

```bash
# To reproduce the results on CVRP1K: 
python main.py --problem_type cvrp  --problem_size 1000 --val_size 100 --revision_lens  20 10 --revision_iters 10 5

# To reproduce the results on CVRP2K: 
python main.py --problem_type cvrp  --problem_size 2000 --val_size 100 --revision_lens 50 20 --revision_iters 15 5

# To reproduce the results on CVRP5K:
python main.py --problem_type cvrp  --problem_size 5000 --val_size 100 --revision_lens 50 20 --revision_iters 15 5

# To reproduce the results on CVRP7K:
python main.py --problem_type cvrp  --problem_size 7000 --val_size 100 --revision_lens 50 20 --revision_iters 15 5

# To reproduce the results on CVRP100K:
python main.py --problem_type cvrp  --problem_size 100000 --val_size 1 --revision_lens 100 50 20 --revision_iters 10 15 5
```

### Evaluation on PCTSP

```bash
# To reproduce the results on PCTSP500: 
python main.py --problem_type pctsp --problem_size 500 --width 1 --n_subset 10 --eval_batch_size 64 --val_size 128

# To reproduce the results on PCTSP1K: 
python main.py --problem_type pctsp --problem_size 1000 --width 1 --n_subset 10 --eval_batch_size 32 --val_size 128

# To reproduce the results on PCTSP10K:
python main.py --problem_type pctsp --problem_size 10000 --width 1 --n_subset 10 --eval_batch_size 2 --val_size 16
```

### Generating data (Stage 1)

To generate the validation dataset before the 1st-stage curriculum learning:
```bash
python generate_data.py
```

### Training (Stage 1)

To train Reviser-100 with multi-distribution SHPP instances:
```bash
python run.py --data_distribution scale --graph_size 100 --n_epochs 200
```

To train Reviser-50, 20, 10:
```bash
--graph_size 50
--graph_size 20
--graph_size 10
```

### Generating data and Training (Stage 2)

Once you obtain the revisers that have finished the 2nd-stage curriculum learning, you can use them to generate training dataset for the next reviser.

```bash
# To generate training dataset for Reviser-100:
python generate_data_RI.py

# To fine-tune Reviser-100:
python run.py --data_distribution scale --RI_train --graph_size 100 --lr_decay 0.99 --RI_path data/RI_train_tsp/500_RI100_seed1235.pt --load_path pretrained/Reviser-stage1/reviser_100/epoch-199.pt --n_epochs 300 --checkpoint_epochs 100

# To generate training dataset for Reviser-50 with Reviser-100:
python generate_data_RG.py --load_path pretrained/Reviser-stage2/reviser_100/epoch-299.pt --data_path data/RI_train_tsp/500_RI100_seed1235.pt --tgt_size 50 --revision_lens 100 --batch_size 50

# To fine-tune Reviser-50:
python run.py --data_distribution scale --RI_train --graph_size 50 --lr_decay 0.99 --RI_path data/RG_train_tsp/RG50.pt --load_path pretrained/Reviser-stage1/reviser_50/epoch-199.pt --n_epochs 300 --checkpoint_epochs 100

# To generate training dataset for Reviser-20 with Reviser-50:
python generate_data_RG.py --load_path pretrained/Reviser-stage2/reviser_50/epoch-299.pt --data_path data/RG_train_tsp/RG50.pt --tgt_size 20 --revision_lens 50 --batch_size 100 

# To fine-tune Reviser-20:
python run.py --data_distribution scale --RI_train --graph_size 20 --lr_decay 0.99 --RI_path data/RG_train_tsp/RG20.pt --load_path pretrained/Reviser-stage1/reviser_20/epoch-199.pt --n_epochs 300 --checkpoint_epochs 100

# To generate training dataset for Reviser-10 with Reviser-20:
python generate_data_RG.py --load_path pretrained/Reviser-stage2/reviser_20/epoch-299.pt --data_path data/RG_train_tsp/RG20.pt --tgt_size 10 --revision_lens 20 --batch_size 100 

# To fine-tune Reviser-10:
python run.py --data_distribution scale --RI_train --graph_size 10 --lr_decay 0.99 --RI_path data/RG_train_tsp/RG10.pt --load_path pretrained/Reviser-stage1/reviser_10/epoch-99.pt --n_epochs 300 --checkpoint_epochs 100
```

## Acknowledgements

* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/alstn12088/LCP
* https://github.com/Spider-scnu/TSP
* https://github.com/jieyibi/AMDKD