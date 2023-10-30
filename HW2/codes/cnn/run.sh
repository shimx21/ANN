#!/bin/bash

LR=1e-3
WD=2e-5
BS=100
EP=100
DR='0.4 0.4'
CCH='128 512'
CKE='5 7'
CST='1 1'
CPD='2 3'
PKE='5 5'
PST='3 4'
PPD='2 2'
DSB=0
DSD=0

NAME="CNN_${CCH}_${CKE}_${CST}_${PKE}_${PST}"

function reset() {
    LR=1e-3
    WD=2e-5
    DR='0.4 0.4'
    CCH='128 512'
    CKE='5 7'
    CST='1 1'
    CPD='2 3'
    PKE='5 5'
    PST='3 4'
    PPD='2 2'
    DSB=0
    DSD=0
}

function train() {
    echo "Testing CNN_${CCH}_${CKE}_${CST}_${PKE}_${PST}_bn${DSB}_dp${DSD}"
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
        --conv_chnl $CCH \
        --conv_kern $CKE \
        --conv_stride $CST \
        --conv_padd $CPD \
        --pool_kern $PKE \
        --pool_stride $PST \
        --pool_padd $PPD \
        --name "$NAME" \
        --wandb=''  \
        --is_train=1 \
        $BNL \
        $DPL \

}

reset
NAME=Test_best
train

echo "Work0: Different Learning Rate"
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2
do
    NAME=CNN_Test_Learning_rate=$LR
    train
done
reset

echo "Work1: Different Weight Decay"
for WD in 0 5e-5 8e-5 1e-4 1.5e-4 2e-4 3e-4
do
    NAME=CNN_Test_Weight_decay=$WD
    train
done
reset

echo "Work2: Different Drop Rate"
for DRL in {1..9}
do
    DR="0.${DRL} 0.${DRL}"
    NAME=CNN_Test_Droprate=$DR
    train
done
reset

echo "Work3: Different Conv Chanel"
for CCH1 in 32 64 128
do
    for CCH2 in 32 64 128 256
    do
        CCH="${CCH1} ${CCH2}"
        NAME=CNN_Test_CCH=$CCH
        train
    done
done
reset

echo "Work4: Different Conv Kernel"
for CKE1 in {3..9..2}
do
    let "CKE2 = CKE1 + 2"
    let "CPD1 = CKE1 / 2"
    let "CPD2 = CKE2 / 2"
    CKE="${CKE1} ${CKE2}"
    CPD="${CPD1} ${CPD2}"
    NAME=CNN_Test_CKE=$CKE
    train
done
reset

echo "Work5: Different Conv Stride"
for CST0 in {1..3}
do
    CST="${CST0} ${CST0}"
    NAME=CNN_Test_CST=$CST
    train
done
reset

echo "Work6: Different Pool Kernel"
for PKE0 in {1..9..2}
do
    let "PPD0 = PKE0 / 2"
    PKE="${PKE0} ${PKE0}"
    PPD="${PPD0} ${PPD0}"
    NAME=CNN_Test_PKE=$PKE
    train
done
reset

echo "Work7: Different Pool Stride"
for PST0 in {1..3}
do
    PST="${PST0} ${PST0}"
    NAME=CNN_Test_PST=$PST
    train
done
reset

echo "Work8: Disable BatchNorm/Dropout"
for DSB in 1 0
do
    for DSD in 1 0
    do
        NAME=CNN_Test_DIS=${DSB}_${DSD}
        train
    done
done
reset