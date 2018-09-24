#!/bin/bash

cores="15"
memory="14000"  # per core
scratch="1000"
gpus="1"
clock="4:00"
model="GeForceGTX1080Ti"
warn="-wt 15 -wa INT"

cmd="bsub
    -n $cores
    -W $clock $output
    $warn
    -R 'select[gpu_model0 == $model] rusage[mem=$memory,scratch=$scratch,ngpus_excl_p=$gpus]'
    $*"
echo $cmd
eval $cmd
