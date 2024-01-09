import os

from clip.clip4clip.clip4clip_adaptory import CLIP4ClipAdaptor, CLIP4CLIP_MODEL_PATH

__all__ = [
    "CLIP4ClipAdaptor",
]

# VATEX FINETUNED InternVideo-MM-L-14
MODEL_URL = "https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/retrieval/vatex/kc4_1e-35e-3_128_8frame/pytorch_model.bin"

if not os.path.exists(CLIP4CLIP_MODEL_PATH):
    os.system(f"cd {os.path.dirname(CLIP4CLIP_MODEL_PATH)} && wget {MODEL_URL}")
