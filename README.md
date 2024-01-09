# 环境
conda activate video_retrieval

# 初次构建
export HUGGINGFACE_TOKEN=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

## ViClip Model
./video_retrieval/tasks/retrieval_task/retrieval.sh init_retrieval.json
## CLIP4Clip Model
./video_retrieval/tasks/retrieval_task/retrieval.sh init_retrieval_clip4clip.json

# resume
./video_retrieval/tasks/retrieval_task/retrieval.sh resume_retrieval.json

