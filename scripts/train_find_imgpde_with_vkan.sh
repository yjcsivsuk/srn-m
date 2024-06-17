#!/bin/bash

pde_find_with_vkan=True
# 要解释的卷积神经网络LeNet
layer_idx=0
# PINN+KAN端到端训练相关参数
with_fu=False  # 固定为False，使vKANPDE:=PINN+KAN
n_layer=5  # pinn隐藏层数量
pd_weight=1.0
pde_weight=1.0
weight_decay=1e-5
epoch=10
lr=1e-3
optim=Adam
# KAN模型相关参数
grid=3
k=3
noise_scale=0.1
noise_scale_base=0.1
symbolic_enabled=True
bias_trainable=True
grid_eps=0.02
sp_trainable=True
sb_trainable=True
device=cpu
gpu=cpu
out_dir=output/find-pde_with_vkan/${n_layer}layers-li${layer_idx}-gs${grid}-opt${optim}-lr${lr}-ep${epoch}

mkdir -p ${out_dir}
nohup python train_img_pde_vkan.py \
    --pde_find_with_vkan ${pde_find_with_vkan} \
    --layer_idx ${layer_idx} \
    --with_fu ${with_fu} \
    --n_layer ${n_layer} \
    --pd_weight ${pd_weight} \
    --pde_weight ${pde_weight} \
    --weight_decay ${weight_decay} \
    --epoch ${epoch} \
    --lr ${lr} \
    --optim ${optim} \
    --grid ${grid} \
    --k ${k} \
    --noise_scale ${noise_scale} \
    --noise_scale_base ${noise_scale_base} \
    --symbolic_enabled ${symbolic_enabled} \
    --bias_trainable ${bias_trainable} \
    --grid_eps ${grid_eps} \
    --sp_trainable ${sp_trainable} \
    --sb_trainable ${sb_trainable} \
    --device ${device} \
    --gpu ${gpu} \
    --out_dir ${out_dir} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid