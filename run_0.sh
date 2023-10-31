# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --arch resnet50 --workers 16 --stage baseline --weight-decay 5e-2 --lr 8e-4 --cos_anneal True --ls True  --opt_name ADAMW --epoch 125  /data/ILSVRC2012 > rn50_bl.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 1e-4 --ls True --p 0.37  --batch-size 512 --lr 0.2 --cos_anneal True --gates 2 --epoch 240 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 100 --grad_mul 5 --reg_w 2.0 /data/ILSVRC2012 > rn50_gate50lmd100.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --ls True --p 0.37  --batch-size 512 --lr 16e-4 --opt_name ADAMW  --cos_anneal True --gates 2 --epoch 240 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 10 --grad_mul 5 --reg_w 2.0 /data/ILSVRC2012 > rn50_gate50lmd10_50.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.37 --batch-size 256 --lr 8e-4 --opt_name ADAMW  --cos_anneal True --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 10 --grad_mul 5 --reg_w 2.0 /p/federatedlearning/data/ILSVRC2012 > rn50_lmd10_50_adam.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.37 --batch-size 256 --lr 8e-4 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 10 --grad_mul 5 --reg_w 2.0 ../model_compress/Data/ILSVRC2012/ > rn50_lmd10_50_adam_1cos_jitter_mixup.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.37  --batch-size 512 --lr 1e-3 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 0 --grad_mul 5 --reg_w 2.0 ../model_compress/Data/ILSVRC2012/ > rn50_lmd0_50_adam_cos_jitter_mixup.txt 2>&1 &


# # help(0.288) -> p = 0.19264250745349115
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.193  --batch-size 1024 --lr 1e-3 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 0 --grad_mul 5 --reg_w 4.0 ../model_compress/Data/ILSVRC2012/ > rn50_lmd0_28_adam_cos_jitter_mixup.txt 2>&1 &

# # help(0.438) -> p = 0.363
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --arch resnet50 --workers 16 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.363  --batch-size 512 --lr 1e-3 --opt_name ADAMW --gates 2 --epoch 245 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 0 --grad_mul 5 --reg_w 4.0 ../model_compress/Data/ILSVRC2012/ > rn50_lmd0_44_adam_cos_jitter_mixup.txt 2>&1 &


# >>> def help(ratio):
# ...     tt = 0.27539
# ...     t = 0.31413
# ...     return (t * ratio - (t-tt)) / tt
# ... 
# >>> 
# >>> help(0.7)
# 0.6577980318820581


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u main.py --arch mobnetv2 --workers 32 --stage train-gate --weight-decay 5e-2 --mix_up True --p 0.6578  --batch-size 1024 --lr 1e-3 --opt_name ADAMW --gates 2 --epoch 305 --start_epoch_hyper 25 --start_epoch_gl 50 --lmd 0 --grad_mul 5 --reg_w 4.0 --pretrained ../model_compress/Data/ILSVRC2012/  > mbv2_lmd0_66_adam_cos_jitter_mixup.txt 2>&1 &

