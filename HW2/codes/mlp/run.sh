#!/bin/bash

LR=1e-3
BS=100
EP=20
DR=0.5
HID=256
DIS=(False False)
NAME="MLP_${DR}_${HID}_${DIS}"

function reset() {
    LR=1e-3
    DR=0.5
    HID=256
    DIS=(False False)
}

function train() {
    echo "Testing MLP_${DR}_${HID}_${DIS}"
    python main.py \
        --batch_size $BS \
        --num_epochs $EP \
        --learning_rate $LR \
        --drop_rate $DR \
        --hid_num $HID\
        --disable_bn ${DIS[0]}\
        --disable_drop ${DIS[1]}\
        --wandb "True"\
        --name "$NAME"
}

reset

# echo "Work0: Different Learning Rate"
# for LR in 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1
# do
#     NAME=MLP_Test_Learning_rate=$LR
#     train
# done
# reset

echo "Work1: Different Drop Rate"
for DR in {1..9}
do
    DR=0.$DR
    NAME=MLP_Test_Droprate=$DR
    train
done
reset

echo "Work2: Different Hidden Layer"
for HID in 32 64 128 256 512 1024
do
    NAME=MLP_Test_HID=$HID
    train
done
reset

echo "Work3: Disable BatchNorm/Dropout"
for DIS_BN in "True" "False"
do
    for DIS_DR in "True" "False"
    do
        DIS=($DIS_BN $DIS_DR)
        NAME=MLP_Test_DIS=$DIS
        train
    done
done
reset