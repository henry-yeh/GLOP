

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




### Evaluation with pretrained model
To produce solution from best M solutions (M=1280), use
```bash
python eval.py --dataset_path data/tsp/tsp20_test_seed1234.pkl --seeder pretrained_LCP/Seeder/seeder_tsp_20/epoch-99.pt --reviser pretrained_LCP/Reviser/reviser_10/epoch-99.pt --softmax_temperature 2 --width 1280 
```

#### LCP star
To use two (can be more, but code must be revised) reviser, use
```bash
python eval.py --dataset_path data/tsp/tsp100_test_seed1234.pkl --seeder pretrained_LCP/Seeder/seeder_tsp_100/epoch-99.pt --reviser pretrained_LCP/Reviser/reviser_20/epoch-99.pt --reviser pretrained_LCP/Reviser/reviser_10/epoch-99.pt --softmax_temperature 2 --width 1280 
```
#### Number of Reviser Iteration (I)

To adjust the number of iteration of reviser, (I=20), use

```bash
--revision_iter1 20
```

For LCP star, use as 

```bash
--revision_iter1 20 --revision_iter2 25
```

#### Revision Length (Note length of sub-problem to revise)

Use as 

```bash
--revision_len1 10
```

For LCP star, use as 

```bash
--revision_len1 20 --revision_len2 10
```

#### Softmax Temperature 

To solve small (N=20,50) problem use high temperature as  

```bash
--softmax_temperature 2
```

To solve large (N=500) problem use low temperature as  

```bash
--softmax_temperature 0.3
```

### Training seeder and reviser

#### Training Seeder

Training Seeder (N=20) with entropy loss (weight=2)

```bash
python run.py --alp 2 --graph_size 20 --policy_mode seeder
```


#### Training Reviser

Training Reviser (N=10, usually smaller sized than seeder)

```bash
python run.py --alp 0 --graph_size 20 --policy_mode reviser --problem local
```

#### Multiple GPU

Defaults training setting is using all GPU devices. Restrict GPU usage as follows:

Two GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py 
```

Single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python run.py 
```


### Other usage

```bash
python run.py -h
python eval.py -h
```



## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib 



