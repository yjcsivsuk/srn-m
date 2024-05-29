#!/bin/bash

gpu=cpu
seed=42

epoch=2000
n_layers=4
layer_idx=0
weight_decay=1e-5
warmup_ratio=0.1
lr=1e-5
clip_norm=0

pde_find=True
with_fu=False
pd_weight=1.0
pde_weight=1.0

out_dir=output/find-pde/${n_layers}layers-li${layer_idx}-clip${clip_norm}-lr${lr}-pw${pd_weight}-pew${pde_weight}-wd${weight_decay}-w${warmup_ratio}
mkdir -p ${out_dir}

nohup python train_img_pde.py \
    --gpu ${gpu} \
    --seed ${seed} \
    --epoch ${epoch} \
    --n_layers ${n_layers} \
    --layer_idx ${layer_idx} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --pde_find ${pde_find} \
    --with_fu ${with_fu} \
    --pd_weight ${pd_weight} \
    --pde_weight ${pde_weight} \
    --lr ${lr} \
    --out_dir ${out_dir} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid