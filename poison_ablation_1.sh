python pretrain.py --poison 5 --unsw --device 0 --log-out poison_ablation/unsw
python pretrain.py --poison 10 --unsw --device 0 --log-out poison_ablation/unsw
python pretrain.py --poison 15 --unsw --device 0 --log-out poison_ablation/unsw
python pretrain.py --poison 20 --unsw --device 0 --log-out poison_ablation/unsw
python pretrain.py --poison 25 --unsw --device 0 --log-out poison_ablation/unsw

python pretrain.py --poison 5 --optc --device 0 --log-out poison_ablation/optc
python pretrain.py --poison 10 --optc --device 0 --log-out poison_ablation/optc
python pretrain.py --poison 15 --optc --device 0 --log-out poison_ablation/optc
python pretrain.py --poison 20 --optc --device 0 --log-out poison_ablation/optc
python pretrain.py --poison 25 --optc --device 0 --log-out poison_ablation/optc

python cls_finetune.py --optc --device 0 --model-fname poison_ablation/optc/rw_bert_optc_5poisoned_tiny.pt --out-dir results/poison_ablation/optc --tag poison5
python cls_finetune.py --optc --device 0 --model-fname poison_ablation/optc/rw_bert_optc_10poisoned_tiny.pt --out-dir results/poison_ablation/optc --tag poison10
python cls_finetune.py --optc --device 0 --model-fname poison_ablation/optc/rw_bert_optc_15poisoned_tiny.pt --out-dir results/poison_ablation/optc --tag poison15
python cls_finetune.py --optc --device 0 --model-fname poison_ablation/optc/rw_bert_optc_20poisoned_tiny.pt --out-dir results/poison_ablation/optc --tag poison20
python cls_finetune.py --optc --device 0 --model-fname poison_ablation/optc/rw_bert_optc_25poisoned_tiny.pt --out-dir results/poison_ablation/optc --tag poison25

python lp_finetune.py --unsw --device 0 --model-fname poison_ablation/unsw/rw_bert_unsw_5poisoned_tiny.pt --out-dir results/poison_ablation/unsw --tag 5poisoned
python lp_finetune.py --unsw --device 0 --model-fname poison_ablation/unsw/rw_bert_unsw_10poisoned_tiny.pt --out-dir results/poison_ablation/unsw --tag 10poisoned
python lp_finetune.py --unsw --device 0 --model-fname poison_ablation/unsw/rw_bert_unsw_15poisoned_tiny.pt --out-dir results/poison_ablation/unsw --tag 15poisoned
python lp_finetune.py --unsw --device 0 --model-fname poison_ablation/unsw/rw_bert_unsw_20poisoned_tiny.pt --out-dir results/poison_ablation/unsw --tag 20poisoned
python lp_finetune.py --unsw --device 0 --model-fname poison_ablation/unsw/rw_bert_unsw_25poisoned_tiny.pt --out-dir results/poison_ablation/unsw --tag 25poisoned