#!/bin/bash

img_pde_find_with_kan=False
pde_find_with_kan=False
pde_find=False

gpu=cpu
seed=42

epoch=1500
n_layers=5
layer_idx=0
weight_decay=1e-5
warmup_ratio=0.1
lr=3e-3
clip_norm=1.0

out_dir=output/img-pde-eql/${n_layers}layers-li${layer_idx}-clip${clip_norm}-lr${lr}-wd${weight_decay}-w${warmup_ratio}
mkdir -p ${out_dir}

nohup python train_img_pde.py \
    --img_pde_find_with_kan ${img_pde_find_with_kan} \
    --pde_find_with_kan ${pde_find_with_kan} \
    --pde_find ${pde_find} \
    --gpu ${gpu} \
    --seed ${seed} \
    --epoch ${epoch} \
    --n_layers ${n_layers} \
    --layer_idx ${layer_idx} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --lr ${lr} \
    --out_dir ${out_dir} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid