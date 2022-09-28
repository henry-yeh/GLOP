# Learning Collaborative Policies to Solve NP-hard Routing Problems

Learhing collaborative policies (LCP) is new problem-solving strategy to tackle NP-hard routing problem such as travelling salesman problem. LCP uses existing competitive model such as Attention Model (AM). We have two main policies: seeder and reviser. Seeder searches full trajectories trained with proposed scaled entropy regularization. Reviser improves seeder's initial feasible candidate solutions in restricted solution space (i.e., partial solution). 



## Paper
This is official PyTorch code for our paper [Learning Collaborative Policies to Solve NP-hard Routing Problems](https://arxiv.org/abs/2110.13987) which has been accepted at [NeurIPS 2021](https://papers.nips.cc/paper/2021), cite our paper as follows:

```
@inproceedings{kim2021learning,
  title={Learning Collaborative Policies to Solve NP-hard Routing Problems},
  author={Kim, Minsu and Park, Jinkyoo and Kim, Joungho},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Thanks to

This code is originally implemented based on  [Attention Model](https://github.com/wouterkool/attention-learn-to-route) , which is source code of the paper   [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) which has been accepted at [ICLR 2019](https://iclr.cc/Conferences/2019), cite as follows:

```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```

Our work designed collaborative polices (seeder and reviser), each policy is parameterized with Attention Model (AM), most of code configuration is same, except:

* New python file containing slightly modified neural architecture for reviser named "nets/attention_local.py".
* Modified "net/attention_model.py" to measure entropy of segment policy (see paper for detail).
* Modified "train.py" to add scaled entropy regularization term. 
* Modified 'sample_many' function in "utils/functions.py" to modify solution design process with collaborative policies. 
* Modified 'eval.py' to modify solution design process.



## Important Remark

* Our work is scheme using two collaborative polices to tackle problem complexity. Therefore, the AM is just example architecture to verify our idea. Please use our idea to state-of-the-art neural combinatorial optimization models to get higher performances.

* This code is focused on TSP. See AM implementation for other TSP-like problems. 

* Our work can be adapted with TSP-like problems. Seeder should be trained with scaled entropy scheme. Reviser can be simply small sized seeder, or can be trained with attention local (only when the start node and the destination node is differ).

## How to Use

### Generating data

To generated random instances (which is benchmark dataset in every NCO papers) use:
```bash
python generate_data.py --problem tsp --name test --seed 1234
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



