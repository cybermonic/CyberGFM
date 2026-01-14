# CyberGFM
Repository for code related to "CyberGFM: Finetuning Graph Foundation Models to Detect Lateral Movement in Enterprise Networks"

Pretrain models using the `pretrain.py` file. 

```
usage: pretrain.py [-h] [--size SIZE] [--device DEVICE]
                   [--optc] [--unsw] [--argus]
                   [--delta DELTA] [--trw]
                   [--tr-size TR_SIZE] [--log-out LOG_OUT]
                   [--n-tokens N_TOKENS]

options:
  -h, --help            show this help message and exit
  --size SIZE           Size of model. Must be in ['tiny', 'mini', 'med', 'baseline']
  --device DEVICE       GPU to use for training (if blank, defaults to 'cpu')
  --optc                Train on the OpTC dataset
  --unsw                Train on the UNSW-NB15 dataset
  --argus               Train on the LANL dataset using features/preprocessing from Argus
  --delta DELTA         Size of time windows to use for temporal random walk sampling 
  --trw                 Use temporal random walks (defaults to static)
  --tr-size TR_SIZE     What percentage of training data to use (for ablation)
  --log-out LOG_OUT     Directory to output log info and checkpoints. Defaults to "."
  --n-tokens N_TOKENS   How many tokens the model sees before training ends 
                        in units of 1e8. E.g. --n-tokens 2 -> 2e8
```

Finetune using either `cls_finetune.py` or `lp_finetune.py`

```
usage: cls_finetune.py [-h] [--size SIZE] [--device DEVICE]
                       [--walk-len WALK_LEN] [--optc]
                       [--unsw] [--argus] [--trw] [--bi]
                       [--from-random]
                       [--model-fname MODEL_FNAME]
                       [--tr-size TR_SIZE] [--tag TAG]

options:
  -h, --help            show this help message and exit
  --size SIZE           Size of model. Must be in ['tiny', 'mini', 'med', 'baseline']
  --device DEVICE       GPU to use for training (if blank, defaults to 'cpu')
  --walk-len WALK_LEN   Length of sampled random walks during finetuning 
  --optc                Train on the OpTC dataset
  --unsw                Train on the UNSW-NB15 dataset
  --argus               Train on the LANL dataset using features/preprocessing from Argus
  --trw                 Use temporal random walks (defaults to static)
  --bi                  
  --from-random
  --model-fname MODEL_FNAME
  --tr-size TR_SIZE
  --tag TAG
```