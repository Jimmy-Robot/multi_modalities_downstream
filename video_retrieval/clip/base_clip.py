from typing import List
from abc import ABC, abstractmethod

import cv2
import torch


class BaseClip(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_video_feature(self, video: cv2.VideoCapture) -> torch.Tensor:
        pass

    @abstractmethod
    def get_text_feature(self, raw_text: str) -> torch.Tensor:
        pass
