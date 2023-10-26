#!/bin/bash
conda activate le
export CUDA_VISIBLE_DEVICES=4

LR=1e-3
DR='0.5 0.5'
BS=100
EP=20
CCH='64 128'
CKE='5 7'
CST='1 1'
CPD='2 3'
PKE='5 5'
PST='3 4'
PPD='2 2'
NAME="CNN_${CCH}_${CKE}_${CST}_${PKE}_${PST}"

function train() {
    echo "Testing CNN_${CCH}_${CKE}_${CST}_${PKE}_${PST}"
    # python main.py \
    #     --batch_size $BS \
    #     --num_epochs $EP \
    #     --learning_rate $LR \
    #     --drop_rate $DR \
    #     --conv_chnl $CCH \
    #     --conv_kern $CKE \
    #     --conv_stride $CST\
    #     --conv_padd $CPD\
    #     --pool_kern $PKE \
    #     --pool_stride $PST \
    #     --pool_padd $PPD\
    #     --wandb \
    #     --name "$NAME"
}

echo "Work1: Different Drop Rate"
for DRL in {1..9}
do
    DR="0.${DRL} 0.${DRL}"
    NAME=CNN_Test_Droprate=$DR
    train
done

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

echo "Work3: Different Conv Kernel"
for CKE1 in {3..9..2}
do
    let "CKE2 = CKE1 + 2"
    let "CPD1 = CKE1 / 2"
    let "CPD2 = CKE2 / 2"
    CKE="${CKE1} ${CKE2}"
    CPD="${CPD1} ${CPD2}"
    NAME=CNN_Test_CCH=$CCH
    train
done

echo "Work4: Different Conv Stride"
for CST0 in {1..3}
do
    CST="${CST0} ${CST0}"
    NAME=CNN_Test_CST=$CST
    train
done

echo "Work5: Different Pool Kernel"
for PKE0 in {3..9..2}
do
    let "PPD0 = PKE0 / 2"
    PKE="${PKE0} ${PKE0}"
    PPD="${PPD0} ${PPD0}"
    NAME=CNN_Test_PKE=$PKE
    train
done

echo "Work6: Different Pool Stride"
for PST0 in {1..3}
do
    PST="${PST0} ${PST0}"
    NAME=CNN_Test_PKE=$PST
    train
done
