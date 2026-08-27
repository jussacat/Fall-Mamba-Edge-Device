import os
import torch
from torch.utils.data import Dataset

class Le2iDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=8, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        
        # Load DUY NHẤT 1 LẦN toàn bộ dữ liệu vào RAM
        self.data_dict = torch.load("/kaggle/working/le2i_all_in_memory.pt", weights_only=True)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Truy xuất O(1) trực tiếp từ RAM, không cần mở/đóng file ổ cứng
        original_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(original_path))[0]
        
        frames_tensor = self.data_dict.get(video_name, torch.zeros((8, 3, 224, 224)))
        label = self.labels[idx]
        
        return frames_tensor, torch.tensor(label, dtype=torch.float32)