import os
import torch
from torch.utils.data import Dataset

class Le2iDataset(Dataset):
    def __init__(self, video_paths, labels, is_train=False):
        self.video_paths = video_paths
        self.labels = labels
        self.is_train = is_train
        
        # Load RAM
        self.data_dict = torch.load("/kaggle/working/le2i_all_in_memory.pt", weights_only=True)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        original_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # Dùng .clone() để tránh làm thay đổi tensor gốc lưu trong RAM khi Augment
        frames_tensor = self.data_dict.get(video_name, torch.zeros((8, 3, 224, 224))).clone()
        label = self.labels[idx]
        
        # Thực hiện Data Augmentation trên GPU/CPU cho tập Train
        if self.is_train:
            # 50% cơ hội lật ngang toàn bộ chuỗi frame (lật theo trục Width - trục cuối cùng)
            if torch.rand(1) < 0.5:
                frames_tensor = frames_tensor.flip(-1)
            # 2. Dark Transformation (Xác suất 30%): Giảm cường độ đặc trưng (mô phỏng thiếu sáng)
            if torch.rand(1) < 0.3:
                # Hệ số ngẫu nhiên từ 0.5 đến 0.8
                dark_factor = 0.5 + 0.3 * torch.rand(1)
                frames_tensor = frames_tensor * dark_factor
            # 3. Frame Masking (Xác suất 30%): Che mất 1-2 khung hình ngẫu nhiên (Temporal Dropout)
            if torch.rand(1) < 0.3:
                # Chọn ngẫu nhiên sẽ xóa 1 hay 2 frame
                num_masked = torch.randint(1, 3, (1,)).item()
                
                # Bốc ngẫu nhiên index của các frame sẽ bị xóa
                mask_indices = torch.randperm(frames_tensor.shape[0])[:num_masked]
                frames_tensor[mask_indices] = 0.0
                
        return frames_tensor, torch.tensor(label, dtype=torch.float32)