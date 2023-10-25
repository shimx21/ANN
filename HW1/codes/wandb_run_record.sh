#!/bin/bash 

WorkFile="run_mlp.py"

# TEST=false
# TEST=true

TEST=$1

if $TEST; 
then
RECORD="True"
PREFIX="echo"
else
RECORD="False"
PREFIX=""
fi 

Jobs=(false false false true)

layers_h0="784 10"
layers_h1="784 100 10"
layers_h2="784 200 50 10"
layers_h3="784 300 100 30 10"

Acts=(Selu Swish Gelu)
Losses=(MSE Softmax Hinge Focal)

Rates=(0.01 0.02 0.05 0.1)
Decays=(0.0 0.2 0.5 0.8)
Moments=(0.0 0.2 0.5 0.8)

runTrain(){
    # hid, rate, moment, act, loss
    if [ $1 -eq 0 ] ; then
    lay=$layers_h0
    else if [ $1 -eq 1 ] ; then
    lay=$layers_h1
    else if [ $1 -eq 2 ] ; then
    lay=$layers_h2
    else 
    lay=$layers_h3
    fi fi fi
    $PREFIX python $WorkFile \
        --name h${1}_rate_${2}_momentum_${3}_act_${4}_loss_${5} --layers $lay --activate $4 --loss $5 \
        --learning_rate $2 --momentum $3 --wandb $RECORD
}

echo ${1}

## RUNS

# Job1: 不同激活函数性能比较
if ${Jobs[0]}; then
for hid in {1..2}; do
for rate in ${Rates[@]:1:2}; do
for moment in ${Moments[@]:0:2}; do
for act in ${Acts[@]}; do
loss=${Losses[2]}
runTrain $hid $rate $moment $act $loss
done done done done
fi

# Job2: 不同损失函数性能比较
if ${Jobs[1]}; then
act=${Acts[0]}
for hid in {1..2}; do
for rate in ${Rates[@]:3:1}; do
for moment in ${Moments[@]:0:2}; do
for loss in ${Losses[@]}; do
runTrain $hid $rate $moment $act $loss
done done done done
fi

# Job3: 损失×激活
if ${Jobs[2]}; then
rate=${Rates[2]}
moment=${Moments[0]}
for hid in {1..1}; do
for act in ${Acts[@]}; do
for loss in ${Losses[@]}; do
runTrain $hid $rate $moment $act $loss
done done done
fi

# Job4: 三隐藏层测试
if ${Jobs[3]}; then
hid=3
rate=${Rates[2]}
moment=${Moments[1]}
act=${Acts[2]}
loss=${Losses[1]}
# for act in ${Acts[@]}; do
# for loss in ${Losses[@]}; do
runTrain $hid $rate $moment $act $loss
# done done
fi