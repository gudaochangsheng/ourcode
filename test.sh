# For CASIA-B
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port XXXXX --nproc_per_node=2 lib/main.py --cfgs ./config/gaitfm.yaml --log_to_file --phase test

# For OUMVLP
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port XXXXX --nproc_per_node=2 lib/main.py --cfgs ./config/gaitfm_OUMVLP.yaml --log_to_file --phase test

