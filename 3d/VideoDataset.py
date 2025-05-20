import os
from torch.utils.data import Dataset
import torchvision.io as io
import torch

class VideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, fps=30, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.max_duration = 1.5  # 只截取前 1.5 秒
        self.fps = fps  # 通常为 30
        self.transform = transform

        self.video_paths = []
        self.labels = []
        self.label_map = {
            'Collide': 0,
            'Contain': 1,
            'Dominoes': 2,
            'Drape': 3,
            'Drop': 4,
            'Link': 5,
            'Roll': 6,
            'Support': 7
        }

        for class_name, label in self.label_map.items():
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.mp4'):
                    self.video_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        # 加载前 1.5 秒视频帧
        end_pts = self.max_duration
        video, _, _ = io.read_video(path, pts_unit='sec', end_pts=end_pts)  # [T, H, W, C]
        
        video = video.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

        # 统一 clip 长度
        if video.shape[1] >= self.clip_len:
            video = video[:, :self.clip_len]
        else:
            pad_len = self.clip_len - video.shape[1]
            pad_tensor = torch.zeros((3, pad_len, video.shape[2], video.shape[3]))
            video = torch.cat([video, pad_tensor], dim=1)

        if self.transform:
            video = self.transform(video)

        return video, label
