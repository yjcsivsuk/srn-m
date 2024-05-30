#!/bin/bash

gpu=cpu
seed=42

img_pde_find_with_kan=True
pde_find_with_kan=False
n_layers=4  # pinn隐藏层的数量
# layers_hidden=[3，3，1]
grid_size=5
spline_order=3
scale_noise=0.1
scale_base=1.0
scale_spline=1.0
grid_eps=0.02
epoch=1000
layer_idx=1
lr=1e-1
optim=AdamW

out_dir=output/img_find-pde_with_kan/${n_layers}layers-li${layer_idx}-gs${grid_size}-opt${optim}-lr${lr}-ep${epoch}
mkdir -p ${out_dir}

nohup python train_img_pde.py \
    --gpu ${gpu} \
    --seed ${seed} \
    --epoch ${epoch} \
    --n_layers ${n_layers} \
    --layer_idx ${layer_idx} \
    --img_pde_find_with_kan ${img_pde_find_with_kan} \
    --pde_find_with_kan ${pde_find_with_kan} \
    --grid_size ${grid_size} \
    --spline_order ${spline_order} \
    --scale_noise ${scale_noise} \
    --scale_base ${scale_base} \
    --scale_spline ${scale_spline} \
    --grid_eps ${grid_eps} \
    --lr ${lr} \
    --out_dir ${out_dir} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid