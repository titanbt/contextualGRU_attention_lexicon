#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/dlnlp/outputs/glove_attention_contextualGRU_non100.out
#PBS -e /home/s1620007/dlnlp/outputs/glove_attention_contextualGRU_non100.in
#PBS -N glove_attention_contextualGRU_non100
#PBS -j oe

cd /home/s1620007/dlnlp

setenv PATH ${PBS_O_PATH}

root="$PWD"

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python $root/main.py -config config/attention_context_s140_sr_glove.cfg -mode attention_contextualGRU