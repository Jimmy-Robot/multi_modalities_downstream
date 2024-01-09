import os

import argparse
import cv2
import torch
import numpy as np
from dataclasses import dataclass

from clip import BaseClip
import clip.utils.video_proc as video_proc_utils
from clip.utils.video_extractor import RawVideoExtractor


from .modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from .modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from .modules.modeling import CLIP4Clip
from .modules.optimization import BertAdam

from clip.vi_clip.ViCLIP.simple_tokenizer import SimpleTokenizer as ViSimpleTokenizer

from utils.logger import logger

CLIP4CLIP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "modules/pytorch_model.bin"
)


@dataclass
class TaskConfig:
    loose_type: bool = True
    linear_patch: str = "2d"
    interaction: str = "no"
    sim_header: str = "meanP"
    clip_evl: bool = True
    pretrained_path: str = CLIP4CLIP_MODEL_PATH
    mergeclip: bool = True
    freeze_layer_num: int = 0
    cross_model: str = "cross-base"
    max_frames: int = 100
    mergeweight: float = 0.5
    max_words: int = 77
    pretrained_clip_name: str = "ViT-L/14"
    slice_framepos: int = 2


class CLIP4ClipAdaptor(BaseClip):
    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")

        # TODO(jimmy): make it configurable
        # so hacky, because CLIP4Clip only accept argsparser result
        args = TaskConfig()
        self.max_words = args.max_words

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        feature_framerate = 1
        image_resolution = 224
        self.frame_order = 0
        self.slice_framepos = args.slice_framepos
        self.max_frames = args.max_frames

        self.video_extractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

        self.tokenizer = ClipTokenizer()
        self.model = CLIP4Clip.from_pretrained(
            args.cross_model, cache_dir=cache_dir, state_dict=None, task_config=args
        )

    def _proc_video(self, video: cv2.VideoCapture):
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        max_video_length = [0]

        # Pair x L x T x 3 x H x W
        video_ts = np.zeros(
            (
                1,
                self.max_frames,
                1,
                3,
                self.video_extractor.size,
                self.video_extractor.size,
            ),
            dtype=np.float32,
        )

        raw_video_data = self.video_extractor.get_video_data(video)
        raw_video_data = raw_video_data["video"]

        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = self.video_extractor.process_raw_data(raw_video_data_clip)
            if self.max_frames < raw_video_slice.shape[0]:
                if self.slice_framepos == 0:
                    video_slice = raw_video_slice[: self.max_frames, ...]
                elif self.slice_framepos == 1:
                    video_slice = raw_video_slice[-self.max_frames :, ...]
                else:
                    sample_indx = np.linspace(
                        0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int
                    )
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            video_slice = self.video_extractor.process_frame_order(
                video_slice, frame_order=self.frame_order
            )

            slice_len = video_slice.shape[0]
            max_video_length[0] = (
                max_video_length[0] if max_video_length[0] > slice_len else slice_len
            )
            if slice_len < 1:
                pass
            else:
                video_ts[0][:slice_len, ...] = video_slice
        else:
            logger.error("_proc_video error.")

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return torch.from_numpy(video_ts), torch.from_numpy(video_mask)

    @torch.no_grad()
    def get_video_feature(self, video: cv2.VideoCapture) -> torch.Tensor:
        logger.debug(
            f"CLIP4ClipAdaptor got video[{video_proc_utils.get_video_info(video)}]"
        )
        video_tensor, video_mask = self._proc_video(video)
        video_mask = video_mask.view(-1, video_mask.shape[-1]).cuda()
        b, bs, ts, channel, h, w = video_tensor.shape
        video_tensor = video_tensor.view(b * bs * ts, channel, h, w).cuda()
        video_frame = bs * ts
        visual_output = self.model.get_visual_output(
            video_tensor, video_mask, shaped=True, video_frame=video_frame
        )
        return visual_output.squeeze(0)  # [B, C]

    @torch.no_grad()
    def get_text_feature(self, raw_text: str) -> torch.Tensor:
        words = self.tokenizer.tokenize(raw_text)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        input_ids = torch.Tensor(input_ids).to(dtype=torch.long)
        input_mask = torch.Tensor(input_mask).to(dtype=torch.long)
        segment_ids = torch.Tensor(segment_ids).to(dtype=torch.long)

        input_ids = input_ids.view(-1, input_ids.shape[-1]).cuda()
        segment_ids = segment_ids.view(-1, segment_ids.shape[-1]).cuda()
        input_mask = input_mask.view(-1, input_mask.shape[-1]).cuda()

        text_features = self.model.get_sequence_output(
            input_ids, segment_ids, input_mask, shaped=True
        )

        return text_features.squeeze(0)  # [B, C]
