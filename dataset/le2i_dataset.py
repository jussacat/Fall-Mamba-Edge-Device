import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

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
        frames = self.data_dict.get(video_name, torch.zeros((8, 3, 224, 224))).clone()
        label = self.labels[idx]
        
        # Thực hiện Data Augmentation trên GPU/CPU cho tập Train
        if self.is_train:
            # 50% cơ hội lật ngang toàn bộ chuỗi frame (lật theo trục Width - trục cuối cùng)
            if torch.rand(1) < 0.5:
                frames = frames.flip(-1)
            # 2. Random Crop & Zoom (Xác suất 40%): Mô phỏng người ở xa hoặc gần camera                                                                                          
            if torch.rand(1) < 0.4:                                                                                                                                              
                # Zoom ngẫu nhiên từ 85% đến 100% kích thước                                                                                                                     
                crop_ratio = 0.85 + 0.15 * torch.rand(1).item()                                                                                                                  
                new_h = int(224 * crop_ratio)                                                                                                                                    
                new_w = int(224 * crop_ratio)                                                                                                                                    
                top = torch.randint(0, 224 - new_h + 1, (1,)).item()                                                                                                             
                left = torch.randint(0, 224 - new_w + 1, (1,)).item()                                                                                                            
                                                                                                                                                                                    
                cropped = frames[:, :, top:top+new_h, left:left+new_w]                                                                                                           
                frames = F.interpolate(cropped, size=(224, 224), mode='bilinear', align_corners=False)                                                                           
                                                                                                                                                                                    
            # 3. Spatial Cutout / Occlusion (Xác suất 35%): Che khuất ngẫu nhiên (bàn, ghế chắn người)                                                                           
            if torch.rand(1) < 0.35:                                                                                                                                             
                occ_size = torch.randint(30, 65, (1,)).item()                                                                                                                    
                occ_top = torch.randint(0, 224 - occ_size, (1,)).item()                                                                                                          
                occ_left = torch.randint(0, 224 - occ_size, (1,)).item()                                                                                                         
                # Che khuất đồng nhất trên toàn bộ chuỗi 8 frames                                                                                                                
                frames[:, :, occ_top:occ_top+occ_size, occ_left:occ_left+occ_size] = 0.0                                                                                         
                                                                                                                                                                                    
            # 4. Temporal Jitter / Speed Perturbation (Xác suất 35%): Thay đổi tốc độ ngã (nhanh/chậm)                                                                           
            if torch.rand(1) < 0.35:                                                                                                                                             
                # Ngẫu nhiên chọn cách lặp hoặc nhảy frame                                                                                                                       
                mode = torch.randint(0, 2, (1,)).item()                                                                                                                          
                if mode == 0:  # Làm chậm (lặp lại 1 frame bất kỳ)                                                                                                               
                    dup_idx = torch.randint(0, 7, (1,)).item()                                                                                                                   
                    indices = list(range(8))                                                                                                                                     
                    indices.insert(dup_idx, dup_idx)                                                                                                                             
                    indices = indices[:8]                                                                                                                                        
                    frames = frames[indices]                                                                                                                                     
                else:          # Temporal Dropout: Làm mờ 1 frame ngẫu nhiên                                                                                                     
                    drop_idx = torch.randint(0, 8, (1,)).item()                                                                                                                  
                    frames[drop_idx] = 0.0                                                                                                                                       
                                                                                                                                                                                    
            # 5. Brightness & Contrast Perturbation (Xác suất 40%): Biến đổi ánh sáng phòng                                                                                      
            if torch.rand(1) < 0.4:                                                                                                                                              
                alpha = 0.85 + 0.3 * torch.rand(1).item()  # Độ tương phản [0.85, 1.15]                                                                                          
                beta = (torch.rand(1).item() - 0.5) * 0.3  # Độ sáng [-0.15, +0.15]                                                                                              
                frames = frames * alpha + beta                                                                                                                                   

            # 6. Sensor Noise (Xác suất 30%): Thêm nhiễu camera an ninh ban đêm
            if torch.rand(1) < 0.3:
                noise = torch.randn_like(frames) * 0.03
                frames = frames + noise
    
        return frames, torch.tensor(label, dtype=torch.float32)