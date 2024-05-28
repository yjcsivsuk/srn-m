#!/bin/bash

GPU=cuda:0

# normal paramters
LR=6e-3
OPTIM=adam
WEIGHT_DECAY=1e-5
REGULAR_TYPE=l2
EPOCH=100
BATCH_SIZE=512
WARMUP_RATIO=0.1

# explained loss
HIDDEN_FN=mse
HIDDEN_WEIGHT=1
OUT_FN=kl
OUT_WEIGHT=0.1
T=2

# for gp/cgp
POP_SIZE=100
ADD_SG=True
LEVELS_BACK=10 
N_ROWS=10
N_COLS=6

SRNET_NAME=MEQL_net
# for EQL, not evolution and grad loss
N_LAYERS=5
EVOLUTION=False
GRAD_LOSS=False

NN_NAME=LeNet
DATASET=data/img/
SRNET_DIR=${SRNET_NAME}/${NN_NAME}/layer${N_LAYERS}-sc-${REGULAR_TYPE}-bs${BATCH_SIZE}-${OPTIM}${LR}-warm${WARMUP_RATIO}-T${T}-h${HIDDEN_FN}${HIDDEN_WEIGHT}-o${OUT_FN}${OUT_WEIGHT}-gl${GRAD_LOSS}-sg${ADD_SG}

mkdir -p output/${SRNET_DIR}

nohup python train.py \
    --train_imgsrnet \
    --nn_name ${NN_NAME} \
    --srnet_name ${SRNET_NAME} \
    --srnet_dir ${SRNET_DIR} \
    --nn_path output/${NN_NAME}/${NN_NAME} \
    --dataset ${DATASET} \
    --lr ${LR} \
    --optim ${OPTIM} \
    --levels_back ${LEVELS_BACK} \
    --n_rows ${N_ROWS} \
    --n_cols ${N_COLS} \
    --evolution ${EVOLUTION} \
    --grad_loss ${GRAD_LOSS} \
    --n_layers ${N_LAYERS} \
    --pop_size ${POP_SIZE} \
    --temperature ${T} \
    --hidden_weight ${HIDDEN_WEIGHT} \
    --out_fn ${OUT_FN} \
    --hidden_fn ${HIDDEN_FN} \
    --out_weight ${OUT_WEIGHT} \
    --weight_decay ${WEIGHT_DECAY} \
    --regular_type ${REGULAR_TYPE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --add_sg_function ${ADD_SG} \
    --epoch ${EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --gpu ${GPU} \
    > output/${SRNET_DIR}/train.log 2>&1 & \
    echo $! > output/${SRNET_DIR}/train.pid