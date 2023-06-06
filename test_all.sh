#!/bin/bash

filepath=output/checkpoint-2-9093/state_dict.bin 

echo $filepath

python3 -m models.tools.end2end_test --model_checkpoint $filepath --dataset library/1029_2
python3 -m models.tools.end2end_test --model_checkpoint $filepath --dataset lab/1013_2
python3 -m models.tools.end2end_test --model_checkpoint $filepath --dataset kitchen/1022_2
python3 -m models.tools.end2end_test --model_checkpoint $filepath --dataset living_room/006
python3 -m models.tools.end2end_test --model_checkpoint $filepath --dataset courtyard/002    courtyard/003

python3 -m models.tools.end2end_test --model_checkpoint $filepath 


