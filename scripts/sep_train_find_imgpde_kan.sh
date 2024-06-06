#!/bin/bash

gpu=cpu
seed=42

# pinn
n_layers=5
opt_pinn=Adam
lr_pinn=5e-3
epoch_pinn=5000

# kan
pinn_train=False
kan_train=True
epoch=500
layer_idx=0
lr=1e-3
optim=LBFGS

pinn_path=output/sep-find-pde/pinn-${n_layers}layers-li${layer_idx}-opt${opt_pinn}-lr${lr_pinn}-ep${epoch_pinn}
out_dir=output/sep-find-pde/kan-li${layer_idx}-opt${optim}-lr${lr}-ep${epoch}-opt_p${opt_pinn}-lr-p${lr_pinn}-ep_p${epoch_pinn}

mkdir -p ${out_dir}

nohup python sep_train_img_pde.py \
    --pinn_train ${pinn_train} \
    --kan_train ${kan_train} \
    --pinn_path ${pinn_path} \
    --gpu ${gpu} \
    --seed ${seed} \
    --epoch ${epoch} \
    --layer_idx ${layer_idx} \
    --optim ${optim} \
    --lr ${lr} \
    --out_dir ${out_dir} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid