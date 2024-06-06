#!/bin/bash

gpu=cpu
seed=42

pinn_train=True
kan_train=False
n_layer=5  # pinn隐藏层的数量
epoch=1000
layer_idx=0
lr=3e-3
optim=AdamW

out_dir=output/sep-find-pde/pinn-${n_layer}layers-li${layer_idx}-opt${optim}-lr${lr}-ep${epoch}
mkdir -p ${out_dir}

nohup python sep_train_img_pde.py \
    --pinn_train ${pinn_train} \
    --kan_train ${kan_train} \
    --gpu ${gpu} \
    --seed ${seed} \
    --epoch ${epoch} \
    --n_layer ${n_layer} \
    --layer_idx ${layer_idx} \
    --optim ${optim} \
    --lr ${lr} \
    --out_dir ${out_dir} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid