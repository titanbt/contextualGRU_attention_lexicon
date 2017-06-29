#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/dlnlp/lexicon_GRU_s140_W2V.out
#PBS -e /home/s1620007/dlnlp/lexicon_GRU_s140_W2V.in
#PBS -N lexicon_GRU_s140_W2V
#PBS -j oe

cd /home/s1620007/dlnlp

setenv PATH ${PBS_O_PATH}

root="$PWD"

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python $root/main.py -config config/attention_context_s140_sr_w2v.cfg -mode lexicon_GRU