class TinyConfig:
    """
    Cấu hình cho mô hình Fall-Mamba phiên bản Tiny.
    Tối ưu hóa để chạy mượt trên GPU T4 của Kaggle.
    """
    
    # 1. Cấu hình Dữ liệu (Dataset)
    img_size = 224          # Kích thước chuẩn của ResNet50
    num_frames = 8          # Số khung hình trích xuất từ mỗi video
    num_classes = 2         # 2 lớp: 1 (Té ngã) và 0 (Bình thường)
    
    # 2. Cấu hình Kiến trúc Mô hình (Model)
    patch_size = 32
    depth = 4              # Độ sâu của mạng Mamba (Bản Tiny dùng 16)
    embed_dim = 128         # Số chiều đặc trưng
    channels = 3            # Ảnh màu RGB (3 kênh)
    
    # Thông số bên trong khối Mamba (ssm_cfg)
    ssm_cfg = {
        'd_state': 16,
        'expand': 2,
        'd_conv': 4,
        'dt_rank': 'auto'
    }
    
    # 3. Cấu hình Huấn luyện (Training Hyperparameters)
    batch_size = 8         # Có thể giảm xuống 8 nếu GPU T4 báo lỗi hết VRAM (Out of Memory)
    epochs = 20             # Số vòng lặp huấn luyện tối đa
    learning_rate = 2e-4    # Tốc độ học khởi tạo
    weight_decay = 0.1      # Hệ số chống Overfitting
    
    # 4. Cấu hình Hệ thống
    num_workers = 2         # Số luồng CPU dùng để nạp dữ liệu (Kaggle hỗ trợ tốt ở mức 2-4)
    seed = 42               # Cố định random seed để có thể tái lập kết quả