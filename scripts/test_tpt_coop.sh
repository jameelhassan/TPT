#!/bin/bash

data_root='./datasets'
coop_weight='weights/coop/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
testsets=$1
# arch=RN50
arch=ViT-B/16
bs=64

/home/jameel.hassan/.conda/envs/maple/bin/python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 --tpt \
--load ${coop_weight}