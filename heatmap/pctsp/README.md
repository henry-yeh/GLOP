### To train partitioners for PCTSP

```bash
# For PCTSP500
python heatmap/pctsp/train.py --problem_size 500 --revision_lens 50 --revision_iters 5

# For PCTSP1K
python heatmap/pctsp/train.py --problem_size 1000 --revision_lens 50 --revision_iters 5

# For PCTSP5K
python heatmap/pctsp/train.py --problem_size 5000 --width 4 --revision_lens 50 --revision_iters 5
```