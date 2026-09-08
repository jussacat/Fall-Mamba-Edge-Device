import os                                                                                                                                                                        
import shutil                                                                                                                                                                   
import cv2                                                                                                                                                                       
import torch                                                                                                                                                                     
import numpy as np                                                                                                                                                               
from tqdm import tqdm                                                                                                                                                            
                                                                                                                                                                                    
# PATH                                                                                                                                                   
RAW_INPUT_DIR = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"                                                                                                             
FORMATTED_DIR = "/kaggle/working/Le2i_Formatted"                                                                                                                                 
CACHE_OUTPUT_FILE = "/kaggle/working/le2i_all_in_memory.pt"                                                                                                                      
NUM_FRAMES = 8                                                                                                                                                                   

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8" 
if os.path.exists(FORMATTED_DIR):                                                                                                                                     
    shutil.rmtree(FORMATTED_DIR)                                                                                                                                    
                                                                                                                                                                                 
# Destination path                                                                                                                                                         
FALL_DIR = os.path.join(FORMATTED_DIR, "Fall")                                                                                                                                   
NORMAL_DIR = os.path.join(FORMATTED_DIR, "Normal")                                                                                                                               
os.makedirs(FALL_DIR, exist_ok=True)                                                                                                                                             
os.makedirs(NORMAL_DIR, exist_ok=True)                                                                                                                                           
                                                                                                                                                                                    
#ImageNet Normalization                                                                                                                                                          
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)                                                                                                                         
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)                                                                                                                          
                                                                                                                                                                                    
print("=" * 65)                                                                                                                                                                  
print("  SCANNING, LABELING & FORMATTING VIDEOS WITH ROOM TAG")                                                                                                          
print("=" * 65)                                                                                                                                                                  
                                                                                                                                                                                    
processed_videos = []                                                                                                                                                            
KNOWN_ROOMS = ["coffee_room", "lecture_room", "office", "home"]

for root, dirs, files in os.walk(RAW_INPUT_DIR):                                                                                                                                 
    for file in files:                                                                                                                                                           
        if file.endswith((".avi", ".mp4")):                                                                                                                                      
            raw_video_path = os.path.join(root, file)                                                                                                                            
            raw_video_name = os.path.splitext(file)[0]                                                                                                                           
                                                                                                                                                                                    
            # Kiem tra nhan Fall dua tren file Annotation (.txt)                                                                                                                 
            annotation_path = os.path.join(root, raw_video_name + ".txt")                                                                                                        
            alt_annotation_path = os.path.join(os.path.dirname(root), "Annotation_files", raw_video_name + ".txt")                                                               
                                                                                                                                                                                    
            is_fall = False                                                                                                                                                      
            if (os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0):
                is_fall = True                                                                                   
            elif (os.path.exists(alt_annotation_path) and os.path.getsize(alt_annotation_path) > 0):                                                                               
                is_fall = True                                                                                                                                                   
                                                                                                                                                                                    
            # Lay ten (room) tu thu muc cha de chia file theo boi canh                                                                                              
            # Eg.: Coffee_room, Home, Office, Lecture_room                                                                                                                     
            path_lower = root.lower().replace(" ", "_")
            room_name = "unknown"
            for r in KNOWN_ROOMS:
                if r in path_lower:
                    room_name = r
                    break                                                                                                                                                                                                               
                                                                                                                                                                                    
            # Name file: [Room]_[OriginalName]                                                                                                                      
            formatted_name = f"{room_name}_{file}"                                                                                                                               
            target_dir = FALL_DIR if is_fall else NORMAL_DIR                                                                                                                     
            target_path = os.path.join(target_dir, formatted_name)                                                                                                               
                                                                                                                                                                                    
            # Copy video                                                                                                                                                         
            shutil.copy(raw_video_path, target_path)                                                                                                                             
            processed_videos.append((target_path, os.path.splitext(formatted_name)[0]))                                                                                          
                                                                                                                                                                                    
fall_count = len(os.listdir(FALL_DIR))                                                                                                                                           
normal_count = len(os.listdir(NORMAL_DIR))                                                                                                                                       
print(f"Total scanned : {len(processed_videos)} videos")                                                                                                                         
print(f"  - Fall      : {fall_count} videos")                                                                                                                                    
print(f"  - Normal    : {normal_count} videos")                                                                                                                                  
                                                                                                                                                                                    
print("\n" + "=" * 65)                                                                                                                                                           
print("  EXTRACTING 8 FRAMES & PACKING INTO IN-MEMORY TENSOR")                                                                                                           
print("=" * 65)                                                                                                                                                                  
                                                                                                                                                                                    
all_data_dict = {}                                                                                                                                                               
                                                                                                                                                                                    
for video_path, video_name in tqdm(processed_videos, desc="Processing Tensors"):                                                                                                 
    cap = cv2.VideoCapture(video_path)                                                                                                                                           
    cap.set(cv2.CAP_PROP_AUDIO_STREAM, -1)                                                                                                                                       
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
                                                                                                                                                                                    
    # Zero padding if video is error or lost frames                                                                                                                              
    if len(frames) > 0:
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].clone())
    else:
        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros((3, 224, 224)))                                                                                                                           
                                                                                                                                  
    all_data_dict[video_name] = torch.stack(frames)                                                                                                                              
                                                                                                                                                                                    
# Save to one cache file                                                                                                                                         
torch.save(all_data_dict, CACHE_OUTPUT_FILE)                                                                                                                                     
print(f"\n[SUCCESS] Packed {len(all_data_dict)} video tensors into: {CACHE_OUTPUT_FILE}")