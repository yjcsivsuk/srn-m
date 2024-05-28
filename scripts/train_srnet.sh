#!/bin/bash

DATASET=529_pollen
LR=1e-3
OPTIM=adam
POP_SIZE=100
EPOCH=2000
NN_NAME=mlp

NN_PATH=output/MLPs/MLP-${DATASET}/mlp/mlp
SRNET_DIR=compared/${NN_NAME}-${DATASET}
mkdir -p output/${SRNET_DIR}

nohup python train.py \
    --train_srnet \
    --nn_name ${NN_NAME} \
    --srnet_dir ${SRNET_DIR} \
    --nn_path ${NN_PATH} \
    --dataset data/${DATASET}.txt \
    --lr ${LR} \
    --optim ${OPTIM} \
    --pop_size ${POP_SIZE} \
    --epoch ${EPOCH} \
    > output/${SRNET_DIR}/train.log 2>&1 & \
    echo $! > output/${SRNET_DIR}/train.pid