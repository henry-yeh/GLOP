

## How to Use

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


### Evaluation with pretrained revisers

```bash
# To reproduce the best result on 10k TSP20: 
python main.py --eval_batch_size 1000  --val_size 10000 --problem_size 20 --revision_lens 20 10 --revision_iters 10 5 --width 80 --decode_strategy sampling --no_aug

# To reproduce the best result on 10k TSP50: 
python main.py --eval_batch_size 500  --val_size 10000 --problem_size 50 --revision_lens 50 20 --revision_iters 30 10 --width 80 --decode_strategy sampling --no_aug


# To reproduce the best result on 10k TSP100: 
python main.py --problem_size 100 --revision_lens 100 50 20 10 --revision_iters 20 10 10 5 --width 80 --eval_batch_size 200 --val_size 10000 --decode_strategy sampling --no_aug

# To reproduce the best result on 128 TSP200: 
python main.py --problem_size 200 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 64 --eval_batch_size 32 --val_size 128 --decode_strategy greedy

# To reproduce the best result on 128 TSP500:
python main.py --problem_size 500 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 64 --eval_batch_size 16 --val_size 128 --decode_strategy greedy

# To reproduce the best result on 128 TSP1000:
python main.py --problem_size 1000 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 64 --eval_batch_size 4 --val_size 128 --decode_strategy greedy

# To reproduce the best result on 16 TSP10k:
python main.py --problem_size 10000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 4 --eval_batch_size 8 --val_size 16 --decode_strategy greedy

# To reproduce the best result on 1 TSP100k:
python main.py --problem_size 100000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1 --eval_batch_size 1 --val_size 1 --decode_strategy greedy

# To conduct cross-distribution evaluation, e.g.:
python main.py --problem_size 100 --revision_lens 100 50 20 10 --revision_iters 20 10 10 5 --width 140 --eval_batch_size 100 --val_size 10000 --decode_strategy sampling --path data/tsp/tsp_uniform100_10000.pkl  --no_aug

```


To reduce the inference duration, try:
```bash
# set
--width 1
```

## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib 



