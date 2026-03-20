(python pretrain.py --poison 5 --argus --device 1 --log-out poison_ablation/lanl; python pretrain.py --poison 10 --argus --device 1 --log-out poison_ablation/lanl) & \
(python pretrain.py --poison 15 --argus --device 2 --log-out poison_ablation/lanl; python pretrain.py --poison 20 --argus --device 2 --log-out poison_ablation/lanl) & \
python pretrain.py --poison 25 --argus --device 3 --log-out poison_ablation/lanl

(python lp_finetune.py --argus --device 1 --model-fname poison_ablation/lanl/rw_bert_lanl14argus_5poisoned_tiny.pt --out-dir results/poison_ablation/lanl/ --tag 5poisoned; python lp_finetune.py --argus --device 0 --model-fname poison_ablation/lanl/rw_bert_lanl14argus_10poisoned_tiny.pt --out-dir results/poison_ablation/lanl/ --tag 10poisoned) & \
(python lp_finetune.py --argus --device 2 --model-fname poison_ablation/lanl/rw_bert_lanl14argus_15poisoned_tiny.pt --out-dir results/poison_ablation/lanl/ --tag 15poisoned; python lp_finetune.py --argus --device 1 --model-fname poison_ablation/lanl/rw_bert_lanl14argus_20poisoned_tiny.pt --out-dir results/poison_ablation/lanl/ --tag 20poisoned) &
python lp_finetune.py --argus --device 3 --model-fname poison_ablation/lanl/rw_bert_lanl14argus_25poisoned_tiny.pt --out-dir results/poison_ablation/lanl/ --tag 25poisoned

