import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class Le2iDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=8, transform=None):
        """
        video_paths: Danh sách (list) đường dẫn tới các file video (.avi, .mp4).
        labels: Danh sách nhãn tương ứng (1 cho Fall, 0 cho Normal).
        num_frames: Số lượng khung hình cần trích xuất từ mỗi video (Bài báo dùng 8).
        transform: Hàm tiền xử lý ảnh (từ transforms.py).
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def _extract_frames(self, video_path):
        """
        Hàm dùng OpenCV để đọc video và lấy ra đúng số lượng num_frames một cách đều đặn.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Nếu video quá ngắn hoặc bị lỗi, tạo mảng đen (zero padding)
        if frame_count == 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        # Tính toán các chỉ số (index) của khung hình cần lấy sao cho rải đều khắp video
        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                # OpenCV đọc ảnh theo hệ màu BGR, cần chuyển sang RGB cho PyTorch
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Fallback nếu lỗi đọc frame
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 1. Trích xuất danh sách các frame ảnh từ video
        frames = self._extract_frames(video_path)

        # 2. Áp dụng phép biến đổi (Resize, Normalize) cho TỪNG frame
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # 3. Gom các frame lại thành 1 Tensor duy nhất
        # Đầu ra của frames đang là list các tensor dạng (C, H, W)
        # Ta dùng torch.stack để xếp chồng chúng lại thành (T, C, H, W) -> T là số frame (8)
        frames_tensor = torch.stack(frames) 
        
        # Trả về tensor dữ liệu và nhãn (chuyển nhãn thành tensor kiểu float)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)