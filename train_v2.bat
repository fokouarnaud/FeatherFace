@echo off
REM Script to launch FeatherFace V2 training with recommended settings

REM Default settings for knowledge distillation training
python train_v2.py ^
    --training_dataset ./data/widerface/train/label.txt ^
    --teacher_model ./weights/mobilenet0.25_Final.pth ^
    --save_folder ./weights/v2/ ^
    --batch_size 32 ^
    --lr 1e-3 ^
    --epochs 400 ^
    --warmup_epochs 5 ^
    --temperature 4.0 ^
    --alpha 0.7 ^
    --feature_weight 0.1 ^
    --mixup_alpha 0.2 ^
    --cutmix_prob 0.5 ^
    --dropblock_prob 0.1 ^
    --dropblock_size 3 ^
    --num_workers 4 ^
    --gpu 0