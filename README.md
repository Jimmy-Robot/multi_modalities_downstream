# 环境
conda activate video_retrieval

# 初次构建
export HUGGINGFACE_TOKEN=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

## ViClip Model
./video_retrieval/tasks/retrieval_task/retrieval.sh ./video_retrieval/tasks/retrieval_task/init_retrieval_clip4clip.json
## CLIP4Clip Model
./video_retrieval/tasks/retrieval_task/retrieval.sh ./video_retrieval/tasks/retrieval_task/init_retrieval_clip4clip.json

# resume
./video_retrieval/tasks/retrieval_task/retrieval.sh ./video_retrieval/tasks/retrieval_task/resume_retrieval.json

