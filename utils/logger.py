import logging
import os
import sys

def setup_logger(log_file="train.log"):
    """
    Khởi tạo Logger. Vừa in ra màn hình (Console), vừa ghi vào file (.log).
    """
    logger = logging.getLogger("FallMamba")
    logger.setLevel(logging.INFO)

    # Chống việc thêm handler nhiều lần nếu hàm được gọi lại
    if not logger.handlers:
        # 1. Ghi ra màn hình Console (để xem trực tiếp trên Kaggle)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 2. Ghi vào file log (để tải về báo cáo)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a') # Ghi nối tiếp (append)
        file_handler.setLevel(logging.INFO)

        # Định dạng dòng chữ in ra (Thời gian - Cấp độ - Nội dung)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger