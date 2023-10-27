#!/bin/bash

LR=1e-3
BS=100
EP=20
DR='0.5 0.5'
CCH='128 256'
CKE='3 5'
CST='2 2'
CPD='1 2'
PKE='3 3'
PST='2 2'
PPD='1 1'
DIS='False False'
NAME="CNN_${CCH}_${CKE}_${CST}_${PKE}_${PST}"

function reset() {
    LR=1e-3
    DR='0.5 0.5'
    CCH='128 256'
    CKE='3 5'
    CST='2 2'
    CPD='1 2'
    PKE='3 3'
    PST='2 2'
    PPD='1 1'
    DIS=(False False)
}

function train() {
    echo "Testing CNN_${CCH}_${CKE}_${CST}_${PKE}_${PST}"
    python main.py \
        --batch_size $BS \
        --num_epochs $EP \
        --learning_rate $LR \
        --drop_rate $DR \
        --conv_chnl $CCH \
        --conv_kern $CKE \
        --conv_stride $CST\
        --conv_padd $CPD\
        --pool_kern $PKE \
        --pool_stride $PST \
        --pool_padd $PPD\
        --disable_bn ${DIS[0]}\
        --disable_drop ${DIS[1]}\
        --wandb "True"\
        --name "$NAME"
}

reset

echo "Work0: Different Learning Rate"
for LR in 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1
do
    NAME=CNN_Test_Learning_rate=$LR
    train
done
reset

echo "Work1: Different Drop Rate"
for DRL in {1..9}
do
    DR="0.${DRL} 0.${DRL}"
    NAME=CNN_Test_Droprate=$DR
    train
done
reset

echo "Work2: Different Conv Chanel"
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

echo "Work3: Different Conv Kernel"
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

echo "Work4: Different Conv Stride"
for CST0 in {1..3}
do
    CST="${CST0} ${CST0}"
    NAME=CNN_Test_CST=$CST
    train
done
reset

echo "Work5: Different Pool Kernel"
for PKE0 in {1..9..2}
do
    let "PPD0 = PKE0 / 2"
    PKE="${PKE0} ${PKE0}"
    PPD="${PPD0} ${PPD0}"
    NAME=CNN_Test_PKE=$PKE
    train
done
reset

echo "Work6: Different Pool Stride"
for PST0 in {1..3}
do
    PST="${PST0} ${PST0}"
    NAME=CNN_Test_PST=$PST
    train
done
reset

echo "Work7: Disable BatchNorm/Dropout"
for DIS_BN in "True" "False"
do
    for DIS_DR in "True" "False"
    do
        DIS=s($DIS_BN $DIS_DR)
        NAME=CNN_Test_DIS=$DIS
        train
    done
done
reset