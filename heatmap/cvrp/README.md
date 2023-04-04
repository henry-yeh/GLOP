### To train partitioners for CVRP

```bash
# For CVRP1K
python heatmap/cvrp/train.py --problem_size 1000 --batch_size 5 --n_epoch 20 --steps_per_epoch 256 --width 20

# For CVRP2K
python heatmap/cvrp/train.py --problem_size 2000 --batch_size 3 --n_epoch 20 --steps_per_epoch 256 --width 10
```