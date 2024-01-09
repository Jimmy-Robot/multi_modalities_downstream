import argparse
import json
import os
from pydantic import ValidationError

from retrieval.ann_based_retrieval import BruteforceRetrieval
from clip.vi_clip import ViCLIPAdapter
from clip.clip4clip import CLIP4ClipAdaptor
from tasks.config import RetrievalConfig
from utils.logger import logger, setup_logger

import torch

torch.cuda.set_device(0)


def get_all_video_paths(dir: str, video_type="mp4"):
    names = os.listdir(dir)

    video_paths = [
        os.path.join(dir, name) for name in names if name.endswith(f".{video_type}")
    ]

    return video_paths


def run_retrieval_task(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    try:
        config = RetrievalConfig(**config_dict)
    except ValidationError as e:
        logger.error(f"config invalid: {e}")
        return

    if config.clip_type == "ViCLIP":
        clip = ViCLIPAdapter()
    elif config.clip_type == "CLIP4Clip":
        clip = CLIP4ClipAdaptor()
    else:
        logger.error(f"can't found: {config.clip_type}")
        return

    if config.searcher_type == "ANN_BF":
        retrieval = BruteforceRetrieval(clip, config.resume_path)
    else:
        logger.error(f"can't found: {config.searcher_type}")
        return

    if config.resume_path is None:
        video_paths = get_all_video_paths(config.video_dir)
        for video_path in video_paths:
            retrieval.insert_vid(video_path, label=video_path)

    with open(config.retrieval_text_file, "r") as f:
        query = json.load(f)

    results = []
    for text in query["retrieval_text"]:
        label, attr, score = retrieval.retrieval_vid(text)
        results.append(
            {
                "text": text,
                "label": label,
                "attr": attr,
                "score": score,
            }
        )

    with open(config.retrieval_result_file_path, "w") as f:
        json.dump(results, f, indent=4)

    if config.serialize_path is not None:
        retrieval.serialize(config.serialize_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retieval Task Arguments.")
    parser.add_argument("--config_path", type=str, help="Path to the config file.")
    args = parser.parse_args()

    setup_logger(os.environ.get("LOG_LEVEL", "debug"))
    run_retrieval_task(args.config_path)
