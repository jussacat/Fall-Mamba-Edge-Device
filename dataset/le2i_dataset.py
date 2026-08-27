import os
import torch
from torch.utils.data import Dataset

class Le2iDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=8, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        # Trỏ trực tiếp vào thư mục chứa Tensor đã tiền xử lý
        self.tensor_dir = "/kaggle/working/Le2i_Tensors"

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Định vị file .pt tương ứng với video
        original_path = self.video_paths[idx]
        category = os.path.basename(os.path.dirname(original_path))
        video_name = os.path.splitext(os.path.basename(original_path))[0]
        
        tensor_path = os.path.join(self.tensor_dir, category, f"{video_name}.pt")
        label = self.labels[idx]

        # Tải thẳng Tensor vào RAM (cực kỳ nhanh)
        if os.path.exists(tensor_path):
            frames_tensor = torch.load(tensor_path, weights_only=True)
        else:
            # Fallback an toàn nếu thiếu file
            frames_tensor = torch.zeros((8, 3, 224, 224))
        
        return frames_tensor, torch.tensor(label, dtype=torch.float32)