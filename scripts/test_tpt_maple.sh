#!/bin/bash

data_root='./datasets'
maple_weight='weights/maple/ori/seed1/MultiModalPromptLearner/model.pth.tar-2'
testsets=$1
# arch=ViT-B/32
arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
<<<<<<< HEAD
-a ${arch} -b ${bs} --gpu 1 --n_ctx 2 \
--tpt --maple --load ${coop_weight}
=======
-a ${arch} -b ${bs} --gpu 0 --n_ctx 2 --maple_depth 3 \
--tpt --maple \
--load ${maple_weight}
>>>>>>> 937d2544605f3f7a5db4b789b2f5234257b49c83
