#!/bin/bash

MODELDIR="/home/ubuntu/Super-SloMo/model/SuperSloMo.ckpt"
VIDEOSFOLDER="/home/ubuntu/Super-SloMo/data/custom/input"
DATASETFOLDER="/home/ubuntu/Super-SloMo/data/custom/output"


#CUDA_VISIBLE_DEVICES='0' python3 -u data/create_dataset.py --videos_folder $VIDEOSFOLDER --dataset_folder $DATASETFOLDER
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset_root $DATASETFOLDER --checkpoint_dir $MODELDIR 2>&1 &


