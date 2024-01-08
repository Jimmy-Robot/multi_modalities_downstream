from abc import ABC, abstractmethod
from typing import List
import os
import pickle

import numpy as np
import torch

from clip import BaseClip


class MetaManager:
    LABEL_LST_SERIALIZE_PATH = "label_lst.pkl"
    ATTR_LST_SERIALIZE_PATH = "attr_lst.pkl"

    def __init__(self, resume_path: str = None):
        self.label_lst = []
        self.attr_lst = []
        if resume_path is not None:
            self.resume(resume_path)

    def resume(self, resume_path: str):
        label_lst_file_path = os.path.join(resume_path, self.LABEL_LST_SERIALIZE_PATH)
        attr_lst_file_path = os.path.join(resume_path, self.ATTR_LST_SERIALIZE_PATH)

        if os.path.exists(label_lst_file_path):
            with open(label_lst_file_path, "rb") as f:
                self.label_lst = pickle.load(f)

        if os.path.exists(attr_lst_file_path):
            with open(attr_lst_file_path, "rb") as f:
                self.attr_lst = pickle.load(f)

    def serialize(self, file_path: str):
        label_lst_file_path = os.path.join(file_path, self.LABEL_LST_SERIALIZE_PATH)
        attr_lst_file_path = os.path.join(file_path, self.ATTR_LST_SERIALIZE_PATH)

        with open(label_lst_file_path, "wb") as f:
            pickle.dump(self.label_lst, f)

        with open(attr_lst_file_path, "wb") as f:
            pickle.dump(self.attr_lst, f)

    def insert(self, label: str, attr: str):
        self.label_lst.append(label)
        self.attr_lst.append(attr)

    def delete(self, label: str):
        if label in self.label_lst:
            index = self.label_lst.index(label)
            self.label_lst.pop(index)
            self.attr_lst.pop(index)
            return index
        return None

    def get_label(self, idx: int):
        return self.label_lst[idx]

    def get_attr(self, idx: int):
        return self.attr_lst[idx]


class BaseRetrieval(ABC):
    def __init__(self, clip: BaseClip, resume_path: str = None):
        self.meta_manager = MetaManager(resume_path=resume_path)
        self.clip = clip
        if resume_path is not None:
            self.resume(resume_path)

    def serialize(self, file_path: str):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.meta_manager.serialize(file_path)
        self.serialize_internal(file_path)

    def insert_vid(self, video_path: str, label: str, attr: str = None, **kwargs):
        self.meta_manager.insert(label, attr)
        self.insert_vid_internal(video_path, **kwargs)

    def delete_vid(
        self,
        label: str,
    ):
        idx = self.meta_manager.delete(label)
        if idx is not None:
            self.delete_vid_internal(idx)

    def retrieval_vid(self, text: str, **kwargs):
        index, score = self.retrieval_vid_internal(text, **kwargs)
        return (
            self.meta_manager.get_label(index),
            self.meta_manager.get_attr(index),
            score,
        )

    @abstractmethod
    def retrieval_vid_internal(self, text: str, **kwargs):
        pass

    @abstractmethod
    def insert_vid_internal(self, video_path: str, **kwargs):
        pass

    @abstractmethod
    def delete_vid_internal(self, idx: int):
        pass

    @abstractmethod
    def resume(self, resume_path: str):
        pass

    @abstractmethod
    def serialize_internal(self, file_path: str):
        pass
