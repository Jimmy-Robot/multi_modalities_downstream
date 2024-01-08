import cv2
import torch

from clip import BaseClip
from .ViCLIP.viclip import ViCLIP
from .ViCLIP.simple_tokenizer import SimpleTokenizer as ViSimpleTokenizer
import clip.utils.video_proc as video_proc_utils

from utils.logger import logger


class ViCLIPAdapter(BaseClip):
    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = ViSimpleTokenizer()
        self.clip = ViCLIP(self.tokenizer).to(self.device)

    @torch.no_grad()
    def get_video_feature(self, video: cv2.VideoCapture) -> torch.Tensor:
        logger.debug(
            f"ViCLIPAdapter got video[{video_proc_utils.get_video_info(video)}]"
        )
        frames = [x for x in video_proc_utils.frame_from_video(video)]
        frames_tensor = video_proc_utils.frames2tensor(frames, device=self.device)
        return self.clip.get_vid_features(frames_tensor)  # [B, C]

    @torch.no_grad()
    def get_text_feature(self, raw_text: str) -> torch.Tensor:
        text_features = self.clip.encode_text(f"{raw_text}").float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features  # [B, C]
