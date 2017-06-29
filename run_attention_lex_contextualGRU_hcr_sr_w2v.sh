#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/dlnlp/outputs/hcr_attention_contextualGRU_W2V.out
#PBS -e /home/s1620007/dlnlp/outputs/hcr_attention_contextualGRU_W2V.in
#PBS -N hcr_acGRU_W2V
#PBS -j oe

cd /home/s1620007/dlnlp

setenv PATH ${PBS_O_PATH}

root="$PWD"

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python $root/main.py -config config/attention_context_hcr_sr_w2v.cfg -mode attention_contextualGRU