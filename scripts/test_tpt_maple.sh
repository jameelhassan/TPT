#!/bin/bash

data_root='./datasets'
coop_weight='weights/maple/vit_b16_c2_ep5_batch4_2ctx_16shots/seed1/MultiModalPromptLearner/model.pth.tar-20'
testsets=$1
# arch=ViT-B/32
arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 1 --n_ctx 2 \
--tpt --maple \
--load ${coop_weight}