#!/bin/bash

# DATASET=kkk1
LR=1e-3
OPTIM=sgd
POP_SIZE=100
EPOCH=2000
NN_NAME=mlp


DATASETS=(feynman0 feynman1 feynman2 feynman3 feynman4 feynman5 kkk0 kkk1 kkk2 kkk3 kkk4 kkk5)
seeds=(12 123 1234 23 234 34 345 45 456 56)

for DATASET in ${DATASETS[*]};
do
    NN_PATH=output/MLPs/MLP-${DATASET}/mlp/mlp
    for seed in ${seeds[*]};
    do
        SRNET_DIR=compared/${NN_NAME}-${DATASET}-${seed}

        mkdir -p output/${SRNET_DIR}
        # echo start dataset=${DATASET}-seed=${seed}
        nohup python train.py \
            --train_srnet \
            --seed ${seed} \
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
    done
    wait
done