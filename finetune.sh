NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' main_finetune.py \
    --accum_iter 4 \
    --batch_size 128 \
    --model vit_small_patch16 \
    --finetune /home/kang_you/mae-main-my/output_dir_mask_ratio0.9/checkpoint-199.pth \
    --nb_classes 3 \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data1/ImageNet/ --output_dir /home/kang_you/mae-main-my/output_SEED_finetune_0.9  --log_dir /home/kang_you/mae-main-my/output_SEED_finetune_0.9 \
# from scrach 72.33
# ratio 0.75 75.62
# ratio 0.90 72.51
# ratio 0.60 72.91
