#!/bin/bash

ws=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ws/../../:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
python $ws/retrieval_task.py --config_path $ws/resume_retrieval.json