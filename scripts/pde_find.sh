#!/bin/bash
sr_class=CGP

problem=burgers
gpu=0

pretrain_epochs=1
pretrain_steps=500
pretrain_lr=1e-3
pretrain_eval_steps=10

pop_size=100
prob=0.4
n_rows=10
n_cols=6
levels_back=20
n_diff_cols=2

batch_size=32
n_epochs=1000
lr=1e-3
optim=adam
eval_steps=10
joint_alpha=0.3

out_dir=output/${problem}/pretrain-joint
mkdir -p ${out_dir}

nohup python train_pde.py \
    --sr_class ${sr_class} \
    --problem ${problem} \
    --gpu ${gpu} \
    --out_dir ${out_dir} \
    --pretrain_pde \
    --pretrain_epochs ${pretrain_epochs} \
    --pretrain_steps ${pretrain_steps} \
    --pretrain_lr ${pretrain_lr} \
    --pretrain_eval_steps ${pretrain_eval_steps} \
    --evolution \
    --pop_size ${pop_size} \
    --prob ${prob} \
    --n_rows ${n_rows} \
    --n_cols ${n_cols} \
    --levels_back ${levels_back} \
    --n_diff_cols ${n_diff_cols} \
    --batch_size ${batch_size} \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --optim ${optim} \
    --eval_steps ${eval_steps} \
    --joint_training \
    --joint_alpha ${joint_alpha} \
    > ${out_dir}/train.log 2>&1 & \
    echo $! > ${out_dir}/train.pid