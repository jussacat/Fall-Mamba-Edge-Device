import os  
import shutil                                                                                                                                                                    
import cv2                                                                                                                                                                       
import torch                                                                                                                                                                     
import numpy as np                                                                                                                                                               
from tqdm import tqdm                                                                                                                                                            
                                                                                                                                                                                    
# Cấu hình đường dẫn dataset mới                                                                                                                                                 
RAW_DATASET_DIR = "/kaggle/input/datasets/faresaljbour/le2i-fall-dataset/Le2i/Le2i"                                                                                              
FALL_SRC_DIR = os.path.join(RAW_DATASET_DIR, "Fall_/Fall")                                                                                                                       
NORMAL_SRC_DIR = os.path.join(RAW_DATASET_DIR, "No_Fall_/No_Fall")                                                                                                               
                                                                                                                                                                                    
FORMATTED_DIR = "/kaggle/working/Le2i_Formatted"                                                                                                                                 
CACHE_OUTPUT_FILE = "/kaggle/working/le2i_all_in_memory.pt"                                                                                                                      
NUM_FRAMES = 8                                                                                                                                                                   
                                                                                                                                                                                    
os.environ["OPENCV_LOG_LEVEL"] = "OFF"                                                                                                                                           
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"                                                                                                                                      
                                                                                                                                                                                    
if os.path.exists(FORMATTED_DIR):                                                                                                                                                
    shutil.rmtree(FORMATTED_DIR)                                                                                                                                                 
                                                                                                                                                                                    
FALL_DEST = os.path.join(FORMATTED_DIR, "Fall")                                                                                                                                  
NORMAL_DEST = os.path.join(FORMATTED_DIR, "Normal")                                                                                                                              
os.makedirs(FALL_DEST, exist_ok=True)                                                                                                                                            
os.makedirs(NORMAL_DEST, exist_ok=True)                                                                                                                                          
                                                                                                                                                                                    
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)                                                                                                                         
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)                                                                                                                          
                                                                                                                                                                                    
print("=" * 50)                                                                                                                                                                  
print("  FORMATTING & COPYING VIDEOS")                                                                                                                                           
print("=" * 50)                                                                                                                                                                  
                                                                                                                                                                                    
processed_videos = []                                                                                                                                                            
                                                                                                                                                                                    
# 1. Quét video Fall (gắn tiền tố fall_ để tránh trùng tên)                                                                                                                      
for f in os.listdir(FALL_SRC_DIR):                                                                                                                                               
    if f.endswith((".avi", ".mp4")):                                                                                                                                             
        src = os.path.join(FALL_SRC_DIR, f)                                                                                                                                      
        unique_name = f"fall_{f}"                                                                                                                                                
        dst = os.path.join(FALL_DEST, unique_name)                                                                                                                               
        shutil.copy(src, dst)                                                                                                                                                    
        processed_videos.append((dst, os.path.splitext(unique_name)[0]))                                                                                                         
                                                                                                                                                                                    
# 2. Quét video Normal (gắn tiền tố normal_ để tránh trùng tên)                                                                                                                  
for f in os.listdir(NORMAL_SRC_DIR):                                                                                                                                             
    if f.endswith((".avi", ".mp4")):                                                                                                                                             
        src = os.path.join(NORMAL_SRC_DIR, f)                                                                                                                                    
        unique_name = f"normal_{f}"                                                                                                                                              
        dst = os.path.join(NORMAL_DEST, unique_name)                                                                                                                             
        shutil.copy(src, dst)                                                                                                                                                    
        processed_videos.append((dst, os.path.splitext(unique_name)[0]))                                                                                                         
                                                                                                                                                                                    
print(f"Tổng số video đã xử lý: {len(processed_videos)}")                                                                                                                        
print(f"  - Fall      : {len(os.listdir(FALL_DEST))} videos")                                                                                                                    
print(f"  - Normal    : {len(os.listdir(NORMAL_DEST))} videos")                                                                                                                  
                                                                                                                                                                                    
print("\n" + "=" * 50)                                                                                                                                                           
print(f"  EXTRACTING {NUM_FRAMES} FRAMES & PACKING INTO IN-MEMORY TENSOR")                                                                                                       
print("=" * 50)                                                                                                                                                                  
                                                                                                                                                                                    
all_data_dict = {}                                                                                                                                                               
                                                                                                                                                                                    
for video_path, video_name in tqdm(processed_videos, desc="Processing Tensors"):                                                                                                 
    cap = cv2.VideoCapture(video_path)                                                                                                                                           
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                                                                                                                         
                                                                                                                                                                                    
    frames = []                                                                                                                                                                  
    if frame_count > 0:                                                                                                                                                          
        target_indices = set(np.linspace(0, frame_count - 1, NUM_FRAMES, dtype=int))                                                                                             
        curr_idx = 0                                                                                                                                                             
        while cap.isOpened() and len(frames) < NUM_FRAMES:                                                                                                                       
            success, frame = cap.read()                                                                                                                                          
            if not success:                                                                                                                                                      
                break                                                                                                                                                            
            if curr_idx in target_indices:                                                                                                                                       
                frame = cv2.resize(frame, (224, 224))                                                                                                                            
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                                                                                   
                tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0                                                                                                    
                tensor = (tensor - mean) / std                                                                                                                                   
                frames.append(tensor)                                                                                                                                            
            curr_idx += 1                                                                                                                                                        
    cap.release()                                                                                                                                                                
                                                                                                                                                                                    
    # Zero padding nếu video bị lỗi hoặc thiếu frames                                                                                                                            
    if len(frames) > 0:                                                                                                                                                          
        while len(frames) < NUM_FRAMES:                                                                                                                                          
            frames.append(frames[-1].clone())                                                                                                                                    
    else:                                                                                                                                                                        
        while len(frames) < NUM_FRAMES:                                                                                                                                          
            frames.append(torch.zeros((3, 224, 224)))                                                                                                                            
                                                                                                                                                                                    
    all_data_dict[video_name] = torch.stack(frames)                                                                                                                              
                                                                                                                                                                                    
torch.save(all_data_dict, CACHE_OUTPUT_FILE)                                                                                                                                     
print(f"\n[SUCCESS] Packed {len(all_data_dict)} video tensors into: {CACHE_OUTPUT_FILE}")