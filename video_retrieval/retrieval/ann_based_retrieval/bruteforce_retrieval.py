from retrieval import BaseRetrieval
from typing import List
import os

import numpy as np
import torch
import cv2

from clip import BaseClip
from utils.logger import logger
from utils.timer import Timer


class BruteforceRetrieval(BaseRetrieval):
    VID_FEATURE_MAT_SERIALIZE_PATH = 'vid_feature_mat.pt'
    def __init__(self, clip: BaseClip, resume_path: str = None):
        self.vid_feature_mat = None
        super().__init__(clip=clip, resume_path=resume_path)

    def insert_vid_internal(self, video_path: str, **kwargs):
        with Timer(f"insert_vid_{video_path}"):
            video = cv2.VideoCapture(video_path)
            video_feature = self.clip.get_video_feature(video)

            if self.vid_feature_mat is None:
                self.vid_feature_mat = video_feature.unsqueeze(0)
            else:
                self.vid_feature_mat = torch.cat([self.vid_feature_mat, video_feature.unsqueeze(0)], dim=0)

    def delete_vid_internal(self, idx: int):
        self.vid_feature_mat = torch.cat([self.vid_feature_mat[:idx], self.vid_feature_mat[idx+1:]], dim=0)

    def retrieval_vid_internal(self, text: str, **kwargs):
        with Timer("retrieval_vid_internal_bf"):
            text_feature = self.clip.get_text_feature(text)

            scores = torch.matmul(self.vid_feature_mat, text_feature.t())

            max_score, max_index = torch.max(scores, dim=0)
            return max_index.item(), max_score.item()

    def resume(self, resume_path: str):
        self.vid_feature_mat = torch.load(os.path.join(resume_path, self.VID_FEATURE_MAT_SERIALIZE_PATH))
        
    def serialize_internal(self, file_path: str):
        torch.save(self.vid_feature_mat, os.path.join(file_path, self.VID_FEATURE_MAT_SERIALIZE_PATH))