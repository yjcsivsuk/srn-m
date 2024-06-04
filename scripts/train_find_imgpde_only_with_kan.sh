#!/bin/bash

# 代替eql的kan网络参数在train_img_pde.py下定义，这里只更改代替pinn的kan网络参数
gpu=cpu
seed=42

pde_find_only_with_kan=True
# layers_hidden=[5,5,1]  # 代替pinn的kan网络的结构
# layers_hidden=[3,3,1]  # 代替eql的kan网络的结构
grid_size=3
spline_order=3
scale_noise=0.1
scale_base=1.0
scale_spline=1.0
grid_eps=0.02
update_grid=True
epoch=1000
layer_idx=0
lr=1e-3
optim=AdamW

out_dir=output/find-pde_only_with_kan/li${layer_idx}-gs${grid_size}-opt${optim}-lr${lr}-ep${epoch}-ug${update_grid}
mkdir -p ${out_dir}

nohup python train_img_pde.py \
    --gpu ${gpu} \
    --seed ${seed} \
    --epoch ${epoch} \
    --layer_idx ${layer_idx} \
    --pde_find_only_with_kan ${pde_find_only_with_kan} \
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