#!/bin/bash

LR=1e-3
WD=3e-4
BS=100
EP=100
DR=0.3
HID=1024
DSB=0
DSD=0

NAME="MLP_${DR}_${HID}_${DSB}"

function reset() {
    LR=1e-3
    WD=3e-4
    DR=0.3
    HID=1024
    DSB=0
    DSD=0
}

function train() {
    echo "Testing MLP_${DR}_${HID}_${DSB}_${DSD}"
    if [ "$DSB" -eq 1 ]; then
        BNL="--disable_bn 1"
    else
        BNL=
    fi
    if [ "$DSD" -eq 1 ]; then
        DPL="--disable_drop 1"
    else
        DPL=
    fi
    python main.py \
        --batch_size $BS \
        --num_epochs $EP \
        --learning_rate $LR \
        --weight_decay $WD \
        --drop_rate $DR \
        --hid_num $HID\
        --wandb='' \
        --is_train=1 \
        --name="$NAME" \
        $BNL  \
        $DPL  \

}

reset
NAME=Test_best
train

echo "Work0: Different Learning Rate"
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2
do
    NAME=MLP_Test_Learning_rate=$LR
    train
done
reset

echo "Work1: Different Weight Decay"
for WD in 0 1e-4 2e-4 3e-4 4e-4 5e-4 6e-4
do
    NAME=MLP_Test_Weight_decay=$WD
    train
done
reset

echo "Work2: Different Drop Rate"
for DR in {1..9}
do
    DR=0.$DR
    NAME=MLP_Test_Droprate=$DR
    train
done
reset

echo "Work3: Different Hidden Layer"
for HID in 128 256 512 1024 2048
do
    NAME=MLP_Test_HID=$HID
    train
done
reset

echo "Work4: Disable BatchNorm/Dropout"
for DSB in 1 0
do
    for DSD in 1 0
    do
        NAME=MLP_Test_STD_DIS=${DSB}_${DSD}
        train
    done
done
reset