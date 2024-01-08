import cv2
import numpy as np
import torch

def frame_from_video(video: cv2.VideoCapture):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    def normalize(data):
        return (data/255.0-v_mean)/v_std
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

def get_video_info(video: cv2.VideoCapture):
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,        
    }