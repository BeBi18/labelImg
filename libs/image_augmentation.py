import os
import random
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton,
                            QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage
import cv2

class AugmentationWidget(QWidget):
    """Widget chứa các tùy chọn augmentation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = self.get_main_window()
        self.init_ui()
        
    def get_main_window(self):
        """Lấy tham chiếu đến MainWindow"""
        parent = self.parent()
        while parent is not None:
            if parent.__class__.__name__ == 'MainWindow':
                return parent
            parent = parent.parent()
        return None
        
    def init_ui(self):
        # Layout chính
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Group các kỹ thuật augmentation
        aug_group = QGroupBox("Kỹ thuật augmentation")
        aug_layout = QVBoxLayout()
        aug_layout.setSpacing(10)
        
        # Rotate
        rotate_layout = QHBoxLayout()
        rotate_layout.setSpacing(10)
        self.rotate_cb = QCheckBox("Xoay ảnh")
        self.rotate_min = QSpinBox()
        self.rotate_min.setRange(-180, 180)
        self.rotate_min.setValue(-10)
        self.rotate_max = QSpinBox() 
        self.rotate_max.setRange(-180, 180)
        self.rotate_max.setValue(10)
        rotate_layout.addWidget(self.rotate_cb)
        rotate_layout.addWidget(QLabel("Góc min:"))
        rotate_layout.addWidget(self.rotate_min)
        rotate_layout.addWidget(QLabel("Góc max:"))
        rotate_layout.addWidget(self.rotate_max)
        aug_layout.addLayout(rotate_layout)
        
        # Flip
        flip_layout = QHBoxLayout()
        flip_layout.setSpacing(10)
        self.flip_cb = QCheckBox("Lật ảnh")
        self.flip_h = QCheckBox("Lật ngang")
        self.flip_h.setChecked(True)
        self.flip_v = QCheckBox("Lật dọc")
        flip_layout.addWidget(self.flip_cb)
        flip_layout.addWidget(self.flip_h)
        flip_layout.addWidget(self.flip_v)
        aug_layout.addLayout(flip_layout)
        
        # Brightness
        bright_layout = QHBoxLayout()
        bright_layout.setSpacing(10)
        self.bright_cb = QCheckBox("Độ sáng")
        self.bright_min = QDoubleSpinBox()
        self.bright_min.setRange(0.5, 2.0)
        self.bright_min.setValue(0.8)
        self.bright_max = QDoubleSpinBox()
        self.bright_max.setRange(0.5, 2.0)
        self.bright_max.setValue(1.2)
        bright_layout.addWidget(self.bright_cb)
        bright_layout.addWidget(QLabel("Min:"))
        bright_layout.addWidget(self.bright_min)
        bright_layout.addWidget(QLabel("Max:"))
        bright_layout.addWidget(self.bright_max)
        aug_layout.addLayout(bright_layout)
        
        # Blur
        blur_layout = QHBoxLayout()
        blur_layout.setSpacing(10)
        self.blur_cb = QCheckBox("Làm mờ")
        self.blur_min = QSpinBox()
        self.blur_min.setRange(1, 10)
        self.blur_min.setValue(1)
        self.blur_max = QSpinBox()
        self.blur_max.setRange(1, 10)
        self.blur_max.setValue(3)
        blur_layout.addWidget(self.blur_cb)
        blur_layout.addWidget(QLabel("Min:"))
        blur_layout.addWidget(self.blur_min)
        blur_layout.addWidget(QLabel("Max:"))
        blur_layout.addWidget(self.blur_max)
        aug_layout.addLayout(blur_layout)
        
        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)
        
        # Group cài đặt số lượng
        settings_group = QGroupBox("Cài đặt số lượng")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)
        
        # Số lượng ảnh mới
        num_layout = QHBoxLayout()
        num_layout.setSpacing(10)
        num_layout.addWidget(QLabel("Số lượng ảnh mới:"))
        self.num_images = QSpinBox()
        self.num_images.setRange(1, 100)
        self.num_images.setValue(5)
        num_layout.addWidget(self.num_images)
        num_layout.addStretch()
        settings_layout.addLayout(num_layout)
        
        # Phần trăm ảnh được chọn
        percent_layout = QHBoxLayout()
        percent_layout.setSpacing(10)
        percent_layout.addWidget(QLabel("Phần trăm ảnh được chọn:"))
        self.percent_images = QSpinBox()
        self.percent_images.setRange(1, 100)
        self.percent_images.setValue(50)
        percent_layout.addWidget(self.percent_images)
        percent_layout.addWidget(QLabel("%"))
        percent_layout.addStretch()
        settings_layout.addLayout(percent_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Nút thực hiện
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.augment_btn = QPushButton("Augment Image")
        self.apply_all_btn = QPushButton("Apply All")
        btn_layout.addWidget(self.augment_btn)
        btn_layout.addWidget(self.apply_all_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        # Kết nối signals
        self.augment_btn.clicked.connect(self.on_augment_clicked)
        self.apply_all_btn.clicked.connect(self.on_apply_all_clicked)
    
    def has_augmentation_enabled(self):
        """Kiểm tra xem có kỹ thuật augmentation nào được bật không"""
        return (self.rotate_cb.isChecked() or 
                self.flip_cb.isChecked() or 
                self.bright_cb.isChecked() or 
                self.blur_cb.isChecked())
        
    def on_augment_clicked(self):
        """Xử lý khi click nút Augment Image"""
        if not self.main_window or not self.main_window.has_labels():
            QMessageBox.warning(self, "Cảnh báo", 
                              "Vui lòng gán nhãn cho ảnh trước khi augmentation!")
            return
        
        # Kiểm tra xem có kỹ thuật augmentation nào được bật không
        if not self.has_augmentation_enabled():
            QMessageBox.warning(self, "Cảnh báo", 
                              "Vui lòng chọn ít nhất một kỹ thuật augmentation!")
            return
            
        # Lấy thư mục lưu
        save_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh")
        if not save_dir:
            return
            
        # Thực hiện augmentation
        self.main_window.augment_current_image(save_dir)
        
    def on_apply_all_clicked(self):
        """Xử lý khi click nút Apply All"""
        if not self.main_window:
            return
        
        # Kiểm tra xem có kỹ thuật augmentation nào được bật không
        if not self.has_augmentation_enabled():
            QMessageBox.warning(self, "Cảnh báo", 
                              "Vui lòng chọn ít nhất một kỹ thuật augmentation!")
            return
            
        # Lấy thư mục lưu
        save_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh")
        if not save_dir:
            return
            
        # Lấy danh sách ảnh có nhãn
        labeled_images = []
        for img_path in self.main_window.m_img_list:
            self.main_window.load_file(img_path)
            if self.main_window.has_labels():
                labeled_images.append(img_path)
                
        if not labeled_images:
            QMessageBox.warning(self, "Cảnh báo", 
                              "Không có ảnh nào có nhãn!")
            return
            
        # Tính số lượng ảnh sẽ được chọn
        percent = self.percent_images.value() / 100.0
        num_images = max(1, int(len(labeled_images) * percent))
        
        # Random chọn ảnh
        selected_images = random.sample(labeled_images, num_images)
        
        # Augment từng ảnh được chọn
        total_augmented = 0
        for img_path in selected_images:
            self.main_window.load_file(img_path)
            self.main_window.augment_current_image(save_dir)
            total_augmented += self.num_images.value()
            
        QMessageBox.information(self, "Thông báo", 
                              f"Đã tạo {total_augmented} ảnh mới từ {num_images} ảnh gốc trong thư mục {save_dir}")

def rotate_image(image, angle):
    """Xoay ảnh một góc angle độ"""
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def flip_image(image, horizontal=False, vertical=False):
    """Lật ảnh theo chiều ngang/dọc"""
    if horizontal:
        image = cv2.flip(image, 1)
    if vertical:
        image = cv2.flip(image, 0)
    return image

def adjust_brightness(image, factor):
    """Điều chỉnh độ sáng của ảnh"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def apply_blur(image, kernel_size):
    """Làm mờ ảnh với kernel size cho trước"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def has_any_augmentation(params):
    """Kiểm tra xem có kỹ thuật augmentation nào được bật không"""
    return (params.get('rotate', False) or 
            params.get('flip', False) or 
            params.get('bright', False) or 
            params.get('blur', False))

def augment_image(image, params):
    """Áp dụng các kỹ thuật augmentation cho ảnh"""
    # Kiểm tra xem có kỹ thuật nào được bật không
    if not has_any_augmentation(params):
        return image.copy()  # Trả về bản copy của ảnh gốc nếu không có augmentation
    
    augmented = image.copy()
    
    # if params.get('rotate', False):
    #     angle = random.uniform(params['rotate_min'], params['rotate_max'])
    #     augmented = rotate_image(augmented, angle)
    #     params['angle'] = angle  # Lưu góc xoay để sử dụng cho bbox
        
    # # Flip
    # if params.get('flip', False):
    #     h_flip = random.random() < 0.5 if params.get('flip_h', False) else False
    #     v_flip = random.random() < 0.5 if params.get('flip_v', False) else False
    #     augmented = flip_image(augmented, h_flip, v_flip)
    #     params['flip_h'] = h_flip  # Lưu trạng thái flip để sử dụng cho bbox
    #     params['flip_v'] = v_flip
        
    # # Brightness
    # if params.get('bright', False):
    #     factor = random.uniform(params['bright_min'], params['bright_max'])
    #     augmented = adjust_brightness(augmented, factor)
        
    # # Blur
    # if params.get('blur', False):
    #     kernel_size = random.randint(params['blur_min'], params['blur_max'])
    #     kernel_size = kernel_size * 2 + 1  # Đảm bảo kernel size là số lẻ
    #     augmented = apply_blur(augmented, kernel_size)

    # Tạo danh sách các phương pháp augmentation được bật
    enabled_methods = []
    if params.get('rotate', False):
        enabled_methods.append('rotate')
    if params.get('flip', False):
        enabled_methods.append('flip')
    if params.get('bright', False):
        enabled_methods.append('bright')
    if params.get('blur', False):
        enabled_methods.append('blur')
        
    # Random số lượng phương pháp sẽ áp dụng (ít nhất 1, nhiều nhất là số phương pháp đã bật)
    num_methods = random.randint(1, len(enabled_methods))
    
    # Random chọn các phương pháp sẽ áp dụng
    selected_methods = random.sample(enabled_methods, num_methods)
    
    # Lưu lại thông tin về các phương pháp đã chọn
    params['applied_methods'] = selected_methods
    
    # Áp dụng các phương pháp đã chọn
    for method in selected_methods:
        if method == 'rotate':
            angle = random.uniform(params['rotate_min'], params['rotate_max'])
            augmented = rotate_image(augmented, angle)
            params['angle'] = angle  # Lưu góc xoay để sử dụng cho bbox
            
        elif method == 'flip':
            h_flip = random.random() < 0.5 if params.get('flip_h', False) else False
            v_flip = random.random() < 0.5 if params.get('flip_v', False) else False
            augmented = flip_image(augmented, h_flip, v_flip)
            params['flip_h'] = h_flip  # Lưu trạng thái flip để sử dụng cho bbox
            params['flip_v'] = v_flip
            
        elif method == 'bright':
            factor = random.uniform(params['bright_min'], params['bright_max'])
            augmented = adjust_brightness(augmented, factor)
            
        elif method == 'blur':
            kernel_size = random.randint(params['blur_min'], params['blur_max'])
            kernel_size = kernel_size * 2 + 1  # Đảm bảo kernel size là số lẻ
            augmented = apply_blur(augmented, kernel_size)
        
    return augmented

def update_bbox(bbox, image_shape, params):
    """Cập nhật tọa độ bounding box sau khi augmentation"""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Chỉ cập nhật bbox cho các phương pháp đã được chọn ngẫu nhiên
    for method in params.get('applied_methods', []):
        if method == 'rotate':
            angle = params.get('angle', 0)
            center = (width/2, height/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Chuyển đổi tọa độ
            points = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            transformed_points = rotation_matrix.dot(points_ones.T).T
            
            x1 = min(transformed_points[:,0])
            y1 = min(transformed_points[:,1])
            x2 = max(transformed_points[:,0])
            y2 = max(transformed_points[:,1])
            
        elif method == 'flip':
            if params.get('flip_h', False):
                x1, x2 = width - x2, width - x1
            if params.get('flip_v', False):
                y1, y2 = height - y2, height - y1
            
    # Đảm bảo tọa độ nằm trong ảnh
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    return [x1, y1, x2, y2] 