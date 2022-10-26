

## How to Use

### Generating data
Once you obtaine the revisers that have finished the 1st-stage curriculum learning, you can use them to generate training dataset for 2nd-stage curriculum learning.

To generate training dataset for reviser100:
```bash
python generate_data_FI.py
```

To generate training dataset for reviser50:
```bash
python generate_data_RG.py --load_path pretrained/Reviser-ft/reviser_100/epoch-299.pt --data_path data/FI_train_tsp/500_FI100_seed1235.pt --tgt_size 50 --revision_lens 100 --batch_size 50
```

To generate training dataset for reviser20:
```bash
python generate_data_RG.py --load_path pretrained/Reviser-ft/reviser_50/epoch-299.pt --data_path data/RG_train_tsp/RG50.pt --tgt_size 20 --revision_lens 50 --batch_size 100 
```

To generate training dataset for reviser10:
```bash
python generate_data_RG.py --load_path pretrained/Reviser-ft/reviser_20/epoch-299.pt --data_path data/RG_train_tsp/RG20.pt --tgt_size 10 --revision_lens 20 --batch_size 100 
```




### Evaluation with pretrained revisers
To solve 10k TSP100:
```bash
python main.py --problem_size 20 --revision_lens 100 20 10 --revision_iters 10 5 --shift_lens 2 2 --aug --aug_shift 5 --eval_batch_size 1000 --val_size 10000
```
```bash
python main.py --problem_size 50 --revision_lens 50 20 10 --revision_iters 25 10 5 --shift_lens 2 2 2 --aug --aug_shift 5 --eval_batch_size 1000 --val_size 10000
```

TSP20 
```bash
python main.py --problem_size 20 --revision_lens 20 10 --revision_iters 10 5 --shift_lens 2 2 --aug --aug_shift 5 --eval_batch_size 1000 --val_size 10000
```


## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib 



