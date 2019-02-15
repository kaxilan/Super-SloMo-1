#!/bin/bash

MODELDIR="/home/ubuntu/Super-SloMo/model"
VIDEOSFOLDER="/home/ubuntu/Super-SloMo/data/custom/input"
DATASETFOLDER="/home/ubuntu/Super-SloMo/data/custom/output"
CHECKPOINT="/home/ubuntu/Super-SloMo/model/SuperSloMo.ckpt"

#CUDA_VISIBLE_DEVICES='0' python3 -u data/create_dataset.py --videos_folder $VIDEOSFOLDER --dataset_folder $DATASETFOLDER
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset_root $DATASETFOLDER --checkpoint_dir $MODELDIR --checkpoint $CHECKPOINT 2>&1 &


