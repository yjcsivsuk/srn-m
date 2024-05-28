#!/bin/bash

#dataset_name=529_pollen
# NN_NAME=mlp
NN_NAME=lenet
# DATASET=data/${dataset_name}.txt
DATASET=data/img
# LOG_DIR=output/MLPs/MLP-${dataset_name}
LOG_DIR=output/LeNet/LeNet

LR=1e-4
OPTIM=adam
EPOCH=3000

mkdir -p ${LOG_DIR}

nohup python train.py \
    --nn_name ${NN_NAME} \
    --dataset ${DATASET} \
    --log_dir ${LOG_DIR} \
    --lr ${LR} \
    --optim ${OPTIM} \
    --epoch ${EPOCH} \
    > ${LOG_DIR}/train.log 2>&1 & \
    echo $! > ${LOG_DIR}/train.pid