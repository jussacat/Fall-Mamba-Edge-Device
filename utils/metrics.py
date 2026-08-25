import torch

class FallMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset các bộ đếm sau mỗi epoch (vòng lặp)"""
        self.tp = 0  # True Positive: Đoán ngã, thực tế ngã
        self.tn = 0  # True Negative: Đoán bình thường, thực tế bình thường
        self.fp = 0  # False Positive: Đoán ngã, thực tế bình thường (Báo động giả)
        self.fn = 0  # False Negative: Đoán bình thường, thực tế ngã (Rất nguy hiểm)

    def update(self, preds, labels):
        """
        Cập nhật kết quả dự đoán so với nhãn gốc (Ground Truth).
        preds: Tensor chứa logit (đầu ra của hàm forward).
        labels: Tensor chứa nhãn (0: Normal, 1: Fall).
        """
        # Chuyển logit thành class dự đoán (0 hoặc 1)
        preds_class = torch.argmax(preds, dim=1)
        labels = labels.int()
        
        # Đếm số lượng theo Confusion Matrix
        self.tp += ((preds_class == 1) & (labels == 1)).sum().item()
        self.tn += ((preds_class == 0) & (labels == 0)).sum().item()
        self.fp += ((preds_class == 1) & (labels == 0)).sum().item()
        self.fn += ((preds_class == 0) & (labels == 1)).sum().item()

    def compute(self):
        """Tính toán các chỉ số khoa học cho khóa luận"""
        # Thêm 1e-8 (số cực nhỏ) để tránh lỗi chia cho 0 nếu mẫu bằng 0
        eps = 1e-8 
        
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        
        # Sensitivity / Recall: Tỉ lệ bắt được cú ngã (Càng cao càng tốt)
        sensitivity = self.tp / (self.tp + self.fn + eps) 
        
        # Specificity: Tỉ lệ không báo động giả
        specificity = self.tn / (self.tn + self.fp + eps)
        
        # Precision: Đoán ngã thì đúng được bao nhiêu phần trăm
        precision = self.tp / (self.tp + self.fp + eps)
        
        # F1-Score: Chỉ số cân bằng giữa Recall và Precision
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
        
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1_score
        }