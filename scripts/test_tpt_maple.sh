#!/bin/bash

data_root='./datasets'
maple_weight='weights/maple/ori/seed1/MultiModalPromptLearner/model.pth.tar-2'
testsets=$1
# arch=ViT-B/32
arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 --n_ctx 2 --maple_depth 3 \
--tpt --maple \
--load ${maple_weight}