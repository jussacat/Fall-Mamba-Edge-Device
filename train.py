import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import random


from configs.tiny_config import TinyConfig
from dataset.le2i_dataset import Le2iDataset
from dataset.transform import get_transforms
from model.fall_mamba import FallMamba
from utils.metrics import FallMetrics
from utils.logger import setup_logger

def get_video_paths_and_labels(data_dir):
    """
    Hàm quét thư mục dataset để lấy đường dẫn video và gán nhãn.
    Giả định cấu trúc: data_dir/Fall/ (nhãn 1) và data_dir/Normal/ (nhãn 0).
    """
    video_paths = []
    labels = []
    
    fall_dir = os.path.join(data_dir, "Fall")
    normal_dir = os.path.join(data_dir, "Normal")
    
    if os.path.exists(fall_dir):
        for file in os.listdir(fall_dir):
            if file.endswith(('.avi', '.mp4')):
                video_paths.append(os.path.join(fall_dir, file))
                labels.append(1)  # 1 là Fall
                
    if os.path.exists(normal_dir):
        for file in os.listdir(normal_dir):
            if file.endswith(('.avi', '.mp4')):
                video_paths.append(os.path.join(normal_dir, file))
                labels.append(0)  # 0 là Normal
                
    return video_paths, labels

def main():
    # 1. Cấu hình tham số dòng lệnh (Command Line Arguments)
    parser = argparse.ArgumentParser(description="Training Fall-Mamba")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--save_path', type=str, default='./working', help="Path saving model and log")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    
    # Initial logger and read config file
    logger = setup_logger(os.path.join(args.save_path, "train.log"))
    cfg = TinyConfig()
    logger.info("Start training Fall-Mamba...")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device used: {device}")

    # Data preprocess
    TEST_ROOM = "office"

    logger.info(f"Scanning data from: {args.data_path}")
    all_paths, all_labels = get_video_paths_and_labels(args.data_path)

    # Split data: TEST_ROOM is the Unseen Room                                                                                                              
    test_paths, test_labels = [], []                                                                                                                                             
    train_val_paths, train_val_labels = [], [] 

    for p, l in zip(all_paths, all_labels):                                                                                                                                      
        file_name = os.path.basename(p).lower()                                                                                                                                  
        if TEST_ROOM in file_name:                                                                                                                                               
            test_paths.append(p)                                                                                                                                                 
            test_labels.append(l)                                                                                                                                                
        else:                                                                                                                                                                    
            train_val_paths.append(p)                                                                                                                                            
            train_val_labels.append(l)

    #Shuffle data
    combined = list(zip(train_val_paths, train_val_labels))
    random.shuffle(combined)
    train_val_paths, train_val_labels = zip(*combined)                                                                                                                           
    train_val_paths, train_val_labels = list(train_val_paths), list(train_val_labels)
    
    #80% Train, 20% Val
    split_idx = int(0.8 * len(train_val_paths))
    train_paths, val_paths = train_val_paths[:split_idx], train_val_paths[split_idx:]
    train_labels, val_labels = train_val_labels[:split_idx], train_val_labels[split_idx:]

    #---------------- Oversampling -------------
    fall_paths = [p for p, l in zip(train_paths, train_labels) if l == 1]
    normal_paths = [p for p, l in zip(train_paths, train_labels) if l == 0]

    fall_count = len(fall_paths)
    normal_count = len(normal_paths)

    if normal_count < fall_count:
        diff = fall_count - normal_count
        oversample_paths = random.choices(normal_paths, k=diff)
        train_paths.extend(oversample_paths)
        train_labels.extend([0] * diff)
        logger.info(f"Oversampling: {diff} Normal videos.")
    if normal_count > fall_count:
        diff = normal_count - fall_count
        oversample_paths = random.choices(fall_paths, k=diff)
        train_paths.extend(oversample_paths)
        train_labels.extend([1] * diff)
        logger.info(f"Oversampling: {diff} Fall videos.")
    
    logger.info(f"Protocol: Leave-One-Room-Out (Test Room: '{TEST_ROOM.upper()}')")                                                                                              
    logger.info(f"Dataset Split: Train={len(train_paths)}, Val={len(val_paths)}, Unseen Test={len(test_paths)}")                                                                 
                                                                                                                    

    train_dataset = Le2iDataset(train_paths, train_labels, is_train=True)                                                                                                        
    val_dataset = Le2iDataset(val_paths, val_labels, is_train=False)                                                                                                             
    test_dataset = Le2iDataset(test_paths, test_labels, is_train=False)                                                                                                          
                                                                                                                                                                                    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)                                                                            
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)                                                                                               
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,    
        pin_memory=True,
        persistent_workers=True,    # Giữ các luồng không bị tắt/mở lại sau mỗi epoch
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # 3. Khởi tạo Mô hình, Loss và Optimizer
    model = FallMamba(
        img_size=cfg.img_size, 
        depth=cfg.depth, 
        embed_dim=cfg.embed_dim,
        num_frames=cfg.num_frames,
        ssm_cfg=cfg.ssm_cfg
    ).to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    metrics = FallMetrics()
    best_f1 = 0.0
    best_epoch = 0
    best_stats = {}
    best_counts = {}

    # 4. Vòng lặp Huấn luyện (Training Loop)
    logger.info("Start training process...")

    scaler = torch.amp.GradScaler('cuda')

    sample_v, sample_l = next(iter(train_loader))
    shape = sample_v.shape
    min_val = sample_v.min().item()
    max_val = sample_v.max().item()
    has_nan = torch.isnan(sample_v).any().item()

    print(
        f"Kiểm tra dữ liệu mẫu: "
        f"Shape={shape}, "
        f"Min={min_val:.3f}, "
        f"Max={max_val:.3f}, "
        f"NaN={has_nan}"
    )

    for epoch in range(cfg.epochs):
        model.train()
        metrics.reset()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.epochs}] Training", leave=False)
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] Loss NaN/Inf in batch {batch_idx}! Skipping this batch...")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            #Gradient 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
  
            train_loss += loss.detach().item()
            metrics.update(outputs, labels)

            if batch_idx % 5 == 0:
                pbar.set_postfix({'Loss': f"{loss.detach().item():.4f}"})
            
        train_stats = metrics.compute()
        avg_train_loss = train_loss / len(train_loader)
        
        # Train Validation
        model.eval()
        metrics.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device, dtype=torch.long)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                metrics.update(outputs, labels)
                
        val_stats = metrics.compute()
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(                                                                                                                                                             
                f"Epoch [{epoch+1:02d}/{cfg.epochs}] │ "                                                                                                                             
                f"Train: Loss={avg_train_loss:.4f}, Acc={train_stats['accuracy']*100:>5.2f}% │ "                                                                                     
                f"Val: Loss={avg_val_loss:.4f}, Acc={val_stats['accuracy']*100:>5.2f}%, F1={val_stats['f1_score']:.4f}, "                                                            
                f"Recall={val_stats['sensitivity']*100:>5.2f}%, Spec={val_stats['specificity']*100:>5.2f}% │ "                                                                       
                f"[TP={metrics.tp}, FN={metrics.fn}, TN={metrics.tn}, FP={metrics.fp}]"                                                                                              
        )

        # Save best model
        if val_stats['f1_score'] > best_f1:                                                                                                                                      
                best_f1 = val_stats['f1_score']                                                                                                                                      
                best_epoch = epoch + 1                                                                                                                                               
                best_stats = val_stats.copy()                                                                                                                                        
                best_counts = {                                                                                                                                                      
                    'tp': metrics.tp,                                                                                                                                                
                    'fn': metrics.fn,                                                                                                                                                
                    'tn': metrics.tn,                                                                                                                                                
                    'fp': metrics.fp                                                                                                                                                 
        }                                                                                                                                                                    
        save_path = os.path.join(args.save_path, "best_fall_mamba.pth")                                                                                                      
        torch.save(model.state_dict(), save_path)                                                                                                                            
        logger.info(f"--> [BEST MODEL SAVED] Epoch {best_epoch:02d} with F1-Score: {best_f1:.4f}")

    # FINAL EVALUATION ON UNSEEN ROOM 
    logger.info("\n" + "=" * 50)
    logger.info(f"       FINAL BENCHMARK ON UNSEEN ROOM: [{TEST_ROOM.upper()}] ({len(test_paths)} VIDEOS)")
    logger.info("=" * 50)
    
    best_model_path = os.path.join(args.save_path, "best_fall_mamba.pth")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint)
    model.eval()
    
    test_metrics = FallMetrics()
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device, dtype=torch.long)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(videos)
            test_metrics.update(outputs, labels)

    test_stats = test_metrics.compute()
    logger.info(f"Unseen Test Accuracy    : {test_stats['accuracy']*100:.2f}% ({test_metrics.tp + test_metrics.tn}/{len(test_paths)} correct)")
    logger.info(f"Unseen Test F1-Score    : {test_stats['f1_score']:.4f}")
    logger.info(f"Unseen Fall Recall (Sens): {test_stats['sensitivity']*100:.2f}% (Rate of detecting real falls in a new room)")
    logger.info(f"Unseen Specificity      : {test_stats['specificity']*100:.2f}% (Rate of avoiding false alarms)")
    logger.info("-" * 50)
    logger.info("CONFUSION MATRIX ON UNSEEN ROOM:")
    logger.info(f"  * True Positive  (TP) : {test_metrics.tp:>2} video(s) -> Fall correctly detected")
    logger.info(f"  * False Negative (FN) : {test_metrics.fn:>2} video(s) -> Fall MISSED (Dangerous!)")
    logger.info(f"  * True Negative  (TN) : {test_metrics.tn:>2} video(s) -> Normal action correctly classified")
    logger.info(f"  * False Positive (FP) : {test_metrics.fp:>2} video(s) -> False Alarm")
    logger.info("=" * 50)
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()