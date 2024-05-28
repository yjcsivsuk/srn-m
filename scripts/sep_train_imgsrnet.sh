#!/bin/bash

GPU=cuda:0

# normal paramters
LR=3e-3
OPTIM=adam
WEIGHT_DECAY=1e-5
EPOCH=5
BATCH_SIZE=512
WARMUP_RATIO=0

# explained loss
HIDDEN_FN=bce
OUT_FN=kl
T=2

# for gp/cgp
POP_SIZE=100
ADD_SG=True
LEVELS_BACK=10 
N_ROWS=10
N_COLS=6

SRNET_NAME=MEQL_net
# for EQL, not evolution and grad loss
EVOLUTION=False
GRAD_LOSS=True

NN_NAME=LeNet
DATASET=data/img/
SRNET_DIR=lbl/${NN_NAME}/${SRNET_NAME}/ep${EPOCH}-bs${BATCH_SIZE}-${OPTIM}${LR}-warm${WARMUP_RATIO}-T${T}-h${HIDDEN_FN}-o${OUT_FN}-gl${GRAD_LOSS}-sg${ADD_SG}

mkdir -p output/${SRNET_DIR}

nohup python sep_train.py \
    --train_imgsrnet \
    --nn_name ${NN_NAME} \
    --srnet_name ${SRNET_NAME} \
    --srnet_dir ${SRNET_DIR} \
    --nn_path output/${NN_NAME}/${NN_NAME} \
    --dataset ${DATASET} \
    --lr ${LR} \
    --optim ${OPTIM} \
    --temperature ${T} \
    --hidden_fn ${HIDDEN_FN} \
    --out_fn ${OUT_FN} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --add_sg_function ${ADD_SG} \
    --evolution ${EVOLUTION} \
    --grad_loss ${GRAD_LOSS} \
    --epoch ${EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --gpu ${GPU} \
    > output/${SRNET_DIR}/train.log 2>&1 & \
    echo $! > output/${SRNET_DIR}/train.pid