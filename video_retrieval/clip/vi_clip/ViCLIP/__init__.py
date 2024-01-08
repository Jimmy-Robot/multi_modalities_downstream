from huggingface_hub import hf_hub_download
import os
from utils.logger import logger

REPO_ID = "OpenGVLab/ViCLIP"
MODEL = "ViClip-InternVid-10M-FLT.pth"
TOKENIZER = "bpe_simple_vocab_16e6.txt.gz"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_model")
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer")

def _has_subfile(dir_path):
    for name in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, name)):
            return True
    return False

if not os.path.exists(MODEL_DIR) or not _has_subfile(MODEL_DIR):
    logger.info(f"download viclip model to {MODEL_DIR}")
    model = hf_hub_download(repo_id=REPO_ID, filename=MODEL, cache_dir=MODEL_DIR, local_dir=MODEL_DIR, token=os.environ.get("HUGGINGFACE_TOKEN", ""))

if not os.path.exists(TOKENIZER_DIR) or not _has_subfile(MODEL_DIR):
    logger.info(f"download viclip tokenizer to {TOKENIZER_DIR}")
    hf_hub_download(repo_id=REPO_ID, filename=TOKENIZER, cache_dir=TOKENIZER_DIR, local_dir=TOKENIZER_DIR, token=os.environ.get("HUGGINGFACE_TOKEN", ""))
