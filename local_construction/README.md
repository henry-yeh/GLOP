# To train local SHPP construction policy

### Generating data (Stage 1)

To generate the validation dataset before the 1st-stage curriculum learning:
```bash
python local_construction/generate_data.py
```

### Training (Stage 1)

To train Reviser-100 with multi-distribution SHPP instances:
```bash
python local_construction/run.py --data_distribution scale --graph_size 100 --n_epochs 200
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
python local_construction/generate_data_RI.py

# To fine-tune Reviser-100:
python local_construction/run.py --data_distribution scale --RI_train --graph_size 100 --lr_decay 0.99 --RI_path data/RI_train_tsp/500_RI100_seed1235.pt --load_path pretrained/Reviser-stage1/reviser_100/epoch-199.pt --n_epochs 300

# To generate training dataset for Reviser-50 with Reviser-100:
python local_construction/generate_data_RG.py --load_path pretrained/Reviser-stage2/reviser_100/epoch-299.pt --data_path data/RI_train_tsp/500_RI100_seed1235.pt --tgt_size 50 --revision_lens 100 --batch_size 50

# To fine-tune Reviser-50:
python local_construction/run.py --data_distribution scale --RI_train --graph_size 50 --lr_decay 0.99 --RI_path data/RG_train_tsp/RG50.pt --load_path pretrained/Reviser-stage1/reviser_50/epoch-199.pt --n_epochs 300
```