import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOHybrid(nn.Module):
    def __init__(self, yolo_model_path, out_dim, n_meta_features=0, n_meta_dim=[512, 128]):
        super(YOLOHybrid, self).__init__()
        self.n_meta_features = n_meta_features
        
        # Load YOLO model từ file .pt (ví dụ: YOLOv8)
        self.yolo = YOLO(yolo_model_path)
        
        # Kiểm tra đầu ra từ YOLO
        dummy_input = torch.randn(1, 3, 640, 640)  # Kích thước ảnh mẫu, điều chỉnh theo yêu cầu của mô hình YOLO
        with torch.no_grad():  # Không tính gradient
            output = self.yolo.model(dummy_input)  # Truyền ảnh qua YOLO
            print("Output type:", type(output))  # In ra kiểu dữ liệu của output
            print("Output shape:", output[0].shape)  # In ra shape của output
        
        # Loại bỏ detection head nếu có (tùy thuộc vào cấu trúc của YOLO)      
        if hasattr(self.yolo.model, 'model'):
            self.yolo.model.model[-1] = nn.Identity()  # Loại bỏ phần detection head
        
        # Xác định kích thước đặc trưng của YOLO (feature_dim)
        # Đảm bảo kích thước này là hợp lệ
        self.feature_dim = output[0].shape[1]  # Lấy số kênh (channels) trong đặc trưng

        # Nếu có metadata, tạo nhánh xử lý metadata
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.SiLU(),  # Sử dụng SiLU (tương tự Swish)
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.SiLU()
            )
            combined_dim = self.feature_dim + n_meta_dim[1]
        else:
            combined_dim = self.feature_dim
        
        self.dropout = nn.Dropout(0.5)
        self.myfc = nn.Linear(combined_dim, out_dim)
    
    def extract(self, x):
        # Trích xuất đặc trưng từ ảnh qua YOLO
        features = self.yolo.model(x)
        # Giả sử output có dạng (batch, C, H, W) -> flatten thành (batch, C*H*W)
        features = torch.flatten(features, start_dim=1)
        return features
    
    def forward(self, x, x_meta=None):
        x_features = self.extract(x)
        # Nếu có metadata, xử lý và ghép nối với đặc trưng ảnh
        if self.n_meta_features > 0 and x_meta is not None:
            x_meta = self.meta(x_meta)
            x_features = torch.cat((x_features, x_meta), dim=1)
        x_features = self.dropout(x_features)
        out = self.myfc(x_features)
        return out
