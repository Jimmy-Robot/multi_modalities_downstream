#!/bin/bash

ws=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ws/../../:$PYTHONPATH

CONFIG=${1:-"init_retrieval.json"}

export CUDA_VISIBLE_DEVICES=0
python $ws/retrieval_task.py --config_path $ws/$CONFIG