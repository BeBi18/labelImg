import os
import random
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton,
                            QFileDialog, QMessageBox, QGroupBox, QDialog,
                            QFrame, QGridLayout, QScrollArea, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch


class PreviewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xem trước Augmentation")
        self.setModal(True)
        self.resize(800, 400)
        
        layout = QHBoxLayout()
        
        # Ảnh gốc
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.original_label)
        
        # Ảnh preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        
        self.setLayout(layout)

    def show_images(self, original_img, preview_img):
        # Chuyển đổi ảnh OpenCV sang QImage
        height, width = original_img.shape[:2]
        bytes_per_line = 3 * width
        q_img_original = QImage(original_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_img_preview = QImage(preview_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Hiển thị ảnh
        self.original_label.setPixmap(QPixmap.fromImage(q_img_original).scaled(380, 380, Qt.KeepAspectRatio))
        self.preview_label.setPixmap(QPixmap.fromImage(q_img_preview).scaled(380, 380, Qt.KeepAspectRatio))
        
        self.exec_()

class AugmentationWidget(QWidget):
    """Widget chứa các tùy chọn augmentation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = self.get_main_window()
        self.preview_dialog = PreviewDialog(self)
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
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Scroll Area cho các kỹ thuật augmentation
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumHeight(400)
        
        # Widget chứa nội dung scroll
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        
        # Group các kỹ thuật augmentation
        aug_group = QGroupBox("Kỹ thuật Augmentation")
        aug_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        aug_layout = QVBoxLayout()
        aug_layout.setSpacing(15)
        aug_layout.setContentsMargins(15, 20, 15, 15)
        
        # Định nghĩa độ rộng cố định cho các cột
        CHECKBOX_WIDTH = 120
        LABEL_WIDTH = 70
        SPINBOX_WIDTH = 70
        BUTTON_WIDTH = 80
        
        # === ROTATE ===
        rotate_frame = QFrame()
        rotate_frame.setFrameStyle(QFrame.StyledPanel)
        rotate_layout = QHBoxLayout(rotate_frame)
        rotate_layout.setSpacing(10)
        rotate_layout.setContentsMargins(10, 10, 10, 10)
        
        self.rotate_cb = QCheckBox("Rotation")
        self.rotate_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        min_label = QLabel("Min:")
        min_label.setFixedWidth(LABEL_WIDTH)
        
        self.rotate_min = QSpinBox()
        self.rotate_min.setRange(-180, 180)
        self.rotate_min.setValue(-10)
        self.rotate_min.setFixedWidth(SPINBOX_WIDTH)
        self.rotate_min.valueChanged.connect(lambda: self.check_min_max(self.rotate_min, self.rotate_max))
        
        max_label = QLabel("Max:")
        max_label.setFixedWidth(LABEL_WIDTH)
        
        self.rotate_max = QSpinBox() 
        self.rotate_max.setRange(-180, 180)
        self.rotate_max.setValue(10)
        self.rotate_max.setFixedWidth(SPINBOX_WIDTH)
        self.rotate_max.valueChanged.connect(lambda: self.check_min_max(self.rotate_min, self.rotate_max))
        
        self.preview_rotate_btn = QPushButton("Preview")
        self.preview_rotate_btn.setFixedWidth(BUTTON_WIDTH)
        
        rotate_layout.addWidget(self.rotate_cb)
        rotate_layout.addWidget(min_label)
        rotate_layout.addWidget(self.rotate_min)
        rotate_layout.addWidget(max_label)
        rotate_layout.addWidget(self.rotate_max)
        rotate_layout.addStretch()
        rotate_layout.addWidget(self.preview_rotate_btn)
        
        aug_layout.addWidget(rotate_frame)
        
        # === FLIP ===
        flip_frame = QFrame()
        flip_frame.setFrameStyle(QFrame.StyledPanel)
        flip_layout = QHBoxLayout(flip_frame)
        flip_layout.setSpacing(10)
        flip_layout.setContentsMargins(10, 10, 10, 10)
        
        self.flip_cb = QCheckBox("Flip")
        self.flip_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        horizontal_label = QLabel("Horizontal:")
        horizontal_label.setFixedWidth(LABEL_WIDTH)
        
        self.flip_h = QCheckBox()
        self.flip_h.setChecked(True)
        self.flip_h.setFixedWidth(SPINBOX_WIDTH)
        
        vertical_label = QLabel("Vertical:")
        vertical_label.setFixedWidth(LABEL_WIDTH)
        
        self.flip_v = QCheckBox()
        self.flip_v.setFixedWidth(SPINBOX_WIDTH)
        
        self.preview_flip_btn = QPushButton("Preview")
        self.preview_flip_btn.setFixedWidth(BUTTON_WIDTH)
        
        flip_layout.addWidget(self.flip_cb)
        flip_layout.addWidget(horizontal_label)
        flip_layout.addWidget(self.flip_h)
        flip_layout.addWidget(vertical_label)
        flip_layout.addWidget(self.flip_v)
        flip_layout.addStretch()
        flip_layout.addWidget(self.preview_flip_btn)
        
        aug_layout.addWidget(flip_frame)
        
        # === BRIGHTNESS ===
        bright_frame = QFrame()
        bright_frame.setFrameStyle(QFrame.StyledPanel)
        bright_layout = QHBoxLayout(bright_frame)
        bright_layout.setSpacing(10)
        bright_layout.setContentsMargins(10, 10, 10, 10)
        
        self.bright_cb = QCheckBox("Brightness")
        self.bright_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        bright_min_label = QLabel("Min:")
        bright_min_label.setFixedWidth(LABEL_WIDTH)
        
        self.bright_min = QDoubleSpinBox()
        self.bright_min.setRange(-99, 99)
        self.bright_min.setValue(0.5)
        self.bright_min.setDecimals(1)
        self.bright_min.setSingleStep(0.1)
        self.bright_min.setFixedWidth(SPINBOX_WIDTH)
        self.bright_min.valueChanged.connect(lambda: self.check_min_max(self.bright_min, self.bright_max))
        
        bright_max_label = QLabel("Max:")
        bright_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.bright_max = QDoubleSpinBox()
        self.bright_max.setRange(-99, 99)
        self.bright_max.setValue(1.2)
        self.bright_max.setDecimals(1)
        self.bright_max.setSingleStep(0.1)
        self.bright_max.setFixedWidth(SPINBOX_WIDTH)
        self.bright_max.valueChanged.connect(lambda: self.check_min_max(self.bright_min, self.bright_max))
        
        self.preview_bright_btn = QPushButton("Preview")
        self.preview_bright_btn.setFixedWidth(BUTTON_WIDTH)
        
        bright_layout.addWidget(self.bright_cb)
        bright_layout.addWidget(bright_min_label)
        bright_layout.addWidget(self.bright_min)
        bright_layout.addWidget(bright_max_label)
        bright_layout.addWidget(self.bright_max)
        bright_layout.addStretch()
        bright_layout.addWidget(self.preview_bright_btn)
        
        aug_layout.addWidget(bright_frame)
        
        # === BLUR ===
        blur_frame = QFrame()
        blur_frame.setFrameStyle(QFrame.StyledPanel)
        blur_layout = QHBoxLayout(blur_frame)
        blur_layout.setSpacing(10)
        blur_layout.setContentsMargins(10, 10, 10, 10)
        
        self.blur_cb = QCheckBox("Blur")
        self.blur_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        blur_label = QLabel("KernelSize:")
        blur_label.setFixedWidth(LABEL_WIDTH)
        
        self.blur_size = QSpinBox()
        self.blur_size.setRange(1, 100)
        self.blur_size.setValue(3)
        self.blur_size.setFixedWidth(SPINBOX_WIDTH)
        
        self.preview_blur_btn = QPushButton("Preview")
        self.preview_blur_btn.setFixedWidth(BUTTON_WIDTH)
        
        blur_layout.addWidget(self.blur_cb)
        blur_layout.addWidget(blur_label)
        blur_layout.addWidget(self.blur_size)
        blur_layout.addStretch()
        blur_layout.addWidget(self.preview_blur_btn)
        
        aug_layout.addWidget(blur_frame)
        
        # === HUE ===
        hue_frame = QFrame()
        hue_frame.setFrameStyle(QFrame.StyledPanel)
        hue_layout = QHBoxLayout(hue_frame)
        hue_layout.setSpacing(10)
        hue_layout.setContentsMargins(10, 10, 10, 10)
        
        self.hue_cb = QCheckBox("Hue")
        self.hue_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        hue_min_label = QLabel("Min:")
        hue_min_label.setFixedWidth(LABEL_WIDTH)
        
        self.hue_min = QSpinBox()
        self.hue_min.setRange(-180, 180)
        self.hue_min.setValue(-10)
        self.hue_min.setFixedWidth(SPINBOX_WIDTH)
        self.hue_min.valueChanged.connect(lambda: self.check_min_max(self.hue_min, self.hue_max))
        
        hue_max_label = QLabel("Max:")
        hue_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.hue_max = QSpinBox()
        self.hue_max.setRange(-180, 180)
        self.hue_max.setValue(10)
        self.hue_max.setFixedWidth(SPINBOX_WIDTH)                                                                                           
        self.hue_max.valueChanged.connect(lambda: self.check_min_max(self.hue_min, self.hue_max))
        
        self.preview_hue_btn = QPushButton("Preview")
        self.preview_hue_btn.setFixedWidth(BUTTON_WIDTH)
        
        hue_layout.addWidget(self.hue_cb)
        hue_layout.addWidget(hue_min_label)
        hue_layout.addWidget(self.hue_min)
        hue_layout.addWidget(hue_max_label)
        hue_layout.addWidget(self.hue_max)
        hue_layout.addStretch()
        hue_layout.addWidget(self.preview_hue_btn)
        
        aug_layout.addWidget(hue_frame)
        
        # === SATURATION ===
        sat_frame = QFrame()
        sat_frame.setFrameStyle(QFrame.StyledPanel)
        sat_layout = QHBoxLayout(sat_frame)
        sat_layout.setSpacing(10)
        sat_layout.setContentsMargins(10, 10, 10, 10)
        
        self.sat_cb = QCheckBox("Saturation")
        self.sat_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        sat_min_label = QLabel("Min:")
        sat_min_label.setFixedWidth(LABEL_WIDTH)
        
        self.sat_min = QDoubleSpinBox()
        self.sat_min.setRange(0, 99)
        self.sat_min.setValue(1)
        self.sat_min.setDecimals(1)
        self.sat_min.setSingleStep(1)
        self.sat_min.setFixedWidth(SPINBOX_WIDTH)
        self.sat_min.valueChanged.connect(lambda: self.check_min_max(self.sat_min, self.sat_max))
        
        sat_max_label = QLabel("Max:")
        sat_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.sat_max = QDoubleSpinBox()
        self.sat_max.setRange(0, 99)
        self.sat_max.setValue(2)
        self.sat_max.setDecimals(1)
        self.sat_max.setSingleStep(1)
        self.sat_max.setFixedWidth(SPINBOX_WIDTH)
        self.sat_max.valueChanged.connect(lambda: self.check_min_max(self.sat_min, self.sat_max))
        
        self.preview_sat_btn = QPushButton("Preview")
        self.preview_sat_btn.setFixedWidth(BUTTON_WIDTH)
        
        sat_layout.addWidget(self.sat_cb)
        sat_layout.addWidget(sat_min_label)
        sat_layout.addWidget(self.sat_min)
        sat_layout.addWidget(sat_max_label)
        sat_layout.addWidget(self.sat_max)
        sat_layout.addStretch()
        sat_layout.addWidget(self.preview_sat_btn)
        
        aug_layout.addWidget(sat_frame)
        
        # === EXPOSURE ===
        exp_frame = QFrame()
        exp_frame.setFrameStyle(QFrame.StyledPanel)
        exp_layout = QHBoxLayout(exp_frame)
        exp_layout.setSpacing(10)
        exp_layout.setContentsMargins(10, 10, 10, 10)
        
        self.exp_cb = QCheckBox("Exposure")
        self.exp_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        exp_min_label = QLabel("Min:")
        exp_min_label.setFixedWidth(LABEL_WIDTH)
        
        self.exp_min = QSpinBox()
        self.exp_min.setRange(-99, 99)
        self.exp_min.setValue(-20)
        self.exp_min.setFixedWidth(SPINBOX_WIDTH)
        self.exp_min.valueChanged.connect(lambda: self.check_min_max(self.exp_min, self.exp_max))
        
        exp_max_label = QLabel("Max:")
        exp_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.exp_max = QSpinBox()
        self.exp_max.setRange(-99, 99)
        self.exp_max.setValue(20)
        self.exp_max.setFixedWidth(SPINBOX_WIDTH)
        self.exp_max.valueChanged.connect(lambda: self.check_min_max(self.exp_min, self.exp_max))
        
        self.preview_exp_btn = QPushButton("Preview")
        self.preview_exp_btn.setFixedWidth(BUTTON_WIDTH)
        
        exp_layout.addWidget(self.exp_cb)
        exp_layout.addWidget(exp_min_label)
        exp_layout.addWidget(self.exp_min)
        exp_layout.addWidget(exp_max_label)
        exp_layout.addWidget(self.exp_max)
        exp_layout.addStretch()
        exp_layout.addWidget(self.preview_exp_btn)
        
        aug_layout.addWidget(exp_frame)
        
        
        # === GRAYSCALE ===
        gray_frame = QFrame()
        gray_frame.setFrameStyle(QFrame.StyledPanel)
        gray_layout = QHBoxLayout(gray_frame)
        gray_layout.setSpacing(10)
        gray_layout.setContentsMargins(10, 10, 10, 10)
        
        self.gray_cb = QCheckBox("Grayscale")
        self.gray_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        self.preview_gray_btn = QPushButton("Preview")
        self.preview_gray_btn.setFixedWidth(BUTTON_WIDTH)
        
        gray_layout.addWidget(self.gray_cb)
        gray_layout.addStretch()
        gray_layout.addWidget(self.preview_gray_btn)
        
        aug_layout.addWidget(gray_frame)

        # === GAUSSIAN NOISE ===
        gaussian_noise_frame = QFrame()
        gaussian_noise_frame.setFrameStyle(QFrame.StyledPanel)
        gaussian_noise_layout = QHBoxLayout(gaussian_noise_frame)
        gaussian_noise_layout.setSpacing(10)
        gaussian_noise_layout.setContentsMargins(10, 10, 10, 10)
        
        self.gaussian_noise_cb = QCheckBox("Gaussian Noise")
        self.gaussian_noise_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        gaussian_noise_label = QLabel("Percent:")
        gaussian_noise_label.setFixedWidth(LABEL_WIDTH)
        
        self.gaussian_noise_percent = QDoubleSpinBox()
        self.gaussian_noise_percent.setRange(0.1, 5.0)
        self.gaussian_noise_percent.setValue(2.0)
        self.gaussian_noise_percent.setDecimals(1)
        self.gaussian_noise_percent.setSingleStep(0.1)
        self.gaussian_noise_percent.setFixedWidth(SPINBOX_WIDTH)
        
        self.preview_gaussian_noise_btn = QPushButton("Preview")
        self.preview_gaussian_noise_btn.setFixedWidth(BUTTON_WIDTH)
        
        gaussian_noise_layout.addWidget(self.gaussian_noise_cb)
        gaussian_noise_layout.addWidget(gaussian_noise_label)
        gaussian_noise_layout.addWidget(self.gaussian_noise_percent)
        gaussian_noise_layout.addStretch()
        gaussian_noise_layout.addWidget(self.preview_gaussian_noise_btn)
        
        aug_layout.addWidget(gaussian_noise_frame)

        # === SALT & PEPPER NOISE ===
        salt_pepper_frame = QFrame()
        salt_pepper_frame.setFrameStyle(QFrame.StyledPanel)
        salt_pepper_layout = QHBoxLayout(salt_pepper_frame)
        salt_pepper_layout.setSpacing(10)
        salt_pepper_layout.setContentsMargins(10, 10, 10, 10)
        
        self.salt_pepper_cb = QCheckBox("Salt Pepper")
        self.salt_pepper_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        salt_pepper_label = QLabel("Percent:")
        salt_pepper_label.setFixedWidth(LABEL_WIDTH)
        
        self.salt_pepper_percent = QDoubleSpinBox()
        self.salt_pepper_percent.setRange(0.1, 5.0)
        self.salt_pepper_percent.setValue(1.0)
        self.salt_pepper_percent.setDecimals(1)
        self.salt_pepper_percent.setSingleStep(0.1)
        self.salt_pepper_percent.setFixedWidth(SPINBOX_WIDTH)
        
        self.preview_salt_pepper_btn = QPushButton("Preview")
        self.preview_salt_pepper_btn.setFixedWidth(BUTTON_WIDTH)
        
        salt_pepper_layout.addWidget(self.salt_pepper_cb)
        salt_pepper_layout.addWidget(salt_pepper_label)
        salt_pepper_layout.addWidget(self.salt_pepper_percent)
        salt_pepper_layout.addStretch()
        salt_pepper_layout.addWidget(self.preview_salt_pepper_btn)
        
        aug_layout.addWidget(salt_pepper_frame)
        
        # === ROTATE 90 ===
        rotate90_frame = QFrame()
        rotate90_frame.setFrameStyle(QFrame.StyledPanel)
        rotate90_layout = QHBoxLayout(rotate90_frame)
        rotate90_layout.setSpacing(10)
        rotate90_layout.setContentsMargins(10, 10, 10, 10)
        
        self.rotate90_cb = QCheckBox("Rotate 90°")
        self.rotate90_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        # Thay thế combobox bằng các checkbox
        self.rotate90_clockwise = QCheckBox("Clockwise")
        self.rotate90_counter = QCheckBox("Counter-clockwise")
        self.rotate90_upside = QCheckBox("Upside-down")
        
        # Đặt giá trị mặc định cho clockwise
        self.rotate90_clockwise.setChecked(True)
        
        self.preview_rotate90_btn = QPushButton("Preview")
        self.preview_rotate90_btn.setFixedWidth(BUTTON_WIDTH)
        
        rotate90_layout.addWidget(self.rotate90_cb)
        rotate90_layout.addWidget(self.rotate90_clockwise)
        rotate90_layout.addWidget(self.rotate90_counter)
        rotate90_layout.addWidget(self.rotate90_upside)
        rotate90_layout.addStretch()
        rotate90_layout.addWidget(self.preview_rotate90_btn)
        
        aug_layout.addWidget(rotate90_frame)
        
        # === CROP ===
        crop_frame = QFrame()
        crop_frame.setFrameStyle(QFrame.StyledPanel)
        crop_layout = QHBoxLayout(crop_frame)
        crop_layout.setSpacing(10)
        crop_layout.setContentsMargins(10, 10, 10, 10)
        
        self.crop_cb = QCheckBox("Crop")
        self.crop_cb.setFixedWidth(CHECKBOX_WIDTH)
        
        crop_min_label = QLabel("Min:")
        crop_min_label.setFixedWidth(LABEL_WIDTH)
        
        self.crop_min = QSpinBox()
        self.crop_min.setRange(0, 99)
        self.crop_min.setValue(10)
        self.crop_min.setFixedWidth(SPINBOX_WIDTH)
        self.crop_min.valueChanged.connect(lambda: self.check_min_max(self.crop_min, self.crop_max))
        
        crop_max_label = QLabel("Max:")
        crop_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.crop_max = QSpinBox()
        self.crop_max.setRange(0, 99)
        self.crop_max.setValue(20)
        self.crop_max.setFixedWidth(SPINBOX_WIDTH)
        self.crop_max.valueChanged.connect(lambda: self.check_min_max(self.crop_min, self.crop_max))
        
        self.preview_crop_btn = QPushButton("Preview")
        self.preview_crop_btn.setFixedWidth(BUTTON_WIDTH)
        
        crop_layout.addWidget(self.crop_cb)
        crop_layout.addWidget(crop_min_label)
        crop_layout.addWidget(self.crop_min)
        crop_layout.addWidget(crop_max_label)
        crop_layout.addWidget(self.crop_max)
        crop_layout.addStretch()
        crop_layout.addWidget(self.preview_crop_btn)
        
        aug_layout.addWidget(crop_frame)
        
        # Thêm stretch để đẩy các phần tử lên trên
        aug_layout.addStretch()
        
        aug_group.setLayout(aug_layout)
        scroll_layout.addWidget(aug_group)
        
        # Set scroll widget
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        # === SETTINGS GROUP ===
        settings_group = QGroupBox("Cài đặt Số lượng")
        settings_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(15)
        settings_layout.setContentsMargins(15, 20, 15, 15)
        
        # Số lượng ảnh mới
        settings_layout.addWidget(QLabel("Số lượng ảnh mới:"),        0, 0, 1, 1)
        self.num_images = QSpinBox()
        self.num_images.setRange(1, 100)
        self.num_images.setValue(5)
        self.num_images.setMinimumWidth(100)
        settings_layout.addWidget(self.num_images,                    0, 1, 1, 1)
        
        # Phần trăm ảnh được chọn
        settings_layout.addWidget(QLabel("Phần trăm ảnh được chọn:"), 1, 0, 1, 1)
        percent_layout = QHBoxLayout()
        self.percent_images = QSpinBox()
        self.percent_images.setRange(1, 100)
        self.percent_images.setValue(50)
        self.percent_images.setMinimumWidth(100)
        percent_layout.addWidget(self.percent_images)
        percent_layout.addWidget(QLabel("%"))
        percent_layout.addStretch()
        settings_layout.addLayout(percent_layout,                     1, 1, 1, 2)
        
        # Thêm stretch cho cột cuối
        settings_layout.setColumnStretch(2, 1)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # === ACTION BUTTONS ===
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(15, 10, 15, 10)
        
        self.augment_btn = QPushButton("Augment Image")
        self.augment_btn.setMinimumHeight(35)
        self.augment_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        
        self.apply_all_btn = QPushButton("Apply All")
        self.apply_all_btn.setMinimumHeight(35)
        self.apply_all_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.augment_btn)
        btn_layout.addWidget(self.apply_all_btn)
        btn_layout.addStretch()
        
        main_layout.addWidget(btn_frame)
        
        self.setLayout(main_layout)
        
        # Kết nối signals
        self.augment_btn.clicked.connect(self.on_augment_clicked)
        self.apply_all_btn.clicked.connect(self.on_apply_all_clicked)
        self.preview_rotate_btn.clicked.connect(self.preview_rotate)
        self.preview_flip_btn.clicked.connect(self.preview_flip)
        self.preview_bright_btn.clicked.connect(self.preview_brightness)
        self.preview_blur_btn.clicked.connect(self.preview_blur)
        self.preview_hue_btn.clicked.connect(self.preview_hue)
        self.preview_sat_btn.clicked.connect(self.preview_saturation)
        self.preview_exp_btn.clicked.connect(self.preview_exposure)
        self.preview_gray_btn.clicked.connect(self.preview_grayscale)
        self.preview_gaussian_noise_btn.clicked.connect(self.preview_gaussian_noise)
        self.preview_salt_pepper_btn.clicked.connect(self.preview_salt_pepper)
        self.preview_rotate90_btn.clicked.connect(self.preview_rotate90)
        self.preview_crop_btn.clicked.connect(self.preview_crop)
    
    def has_augmentation_enabled(self):
        """Kiểm tra xem có kỹ thuật augmentation nào được bật không"""
        return (self.rotate_cb.isChecked() or 
                self.flip_cb.isChecked() or 
                self.bright_cb.isChecked() or 
                self.blur_cb.isChecked() or
                self.hue_cb.isChecked() or
                self.sat_cb.isChecked() or
                self.exp_cb.isChecked() or
                self.gray_cb.isChecked() or
                self.gaussian_noise_cb.isChecked() or
                self.salt_pepper_cb.isChecked() or
                self.crop_cb.isChecked() or
                self.rotate90_cb.isChecked())
        
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

    def get_current_image(self):
        """Lấy ảnh hiện tại từ main window"""
        if not self.main_window or not self.main_window.original_image:
            return None
            
        # Chuyển QImage sang numpy array
        image = self.main_window.original_image
        width = image.width()
        height = image.height()
        
        # Chuyển đổi QImage sang numpy array dựa trên format của ảnh
        if image.format() == QImage.Format_RGB32:
            ptr = image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        elif image.format() == QImage.Format_ARGB32:
            ptr = image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            arr = arr[:, :, [2, 1, 0]]  # A,R,G,B -> B,G,R
        elif image.format() == QImage.Format_RGB888:
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            # Chuyển sang RGB nếu là grayscale
            image = image.convertToFormat(QImage.Format_RGB888)
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
        return arr

    def preview_rotate(self):
        """Xem trước hiệu ứng xoay"""
        if not self.rotate_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn xoay ảnh!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với góc min và max
        img_min = rotate_image(img.copy(), self.rotate_min.value())
        img_max = rotate_image(img.copy(), self.rotate_max.value())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_min)
        self.preview_dialog.show_images(img, img_max)

    def preview_flip(self):
        """Xem trước hiệu ứng lật"""
        if not self.flip_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn lật ảnh!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với các tùy chọn lật
        if self.flip_h.isChecked():
            img_h = flip_image(img.copy(), horizontal=True)
            self.preview_dialog.show_images(img, img_h)
            
        if self.flip_v.isChecked():
            img_v = flip_image(img.copy(), vertical=True)
            self.preview_dialog.show_images(img, img_v)

    def preview_brightness(self):
        """Xem trước hiệu ứng độ sáng"""
        if not self.bright_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn độ sáng!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với độ sáng min và max
        img_min = adjust_brightness(img.copy(), self.bright_min.value())
        img_max = adjust_brightness(img.copy(), self.bright_max.value())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_min)
        self.preview_dialog.show_images(img, img_max)

    def preview_blur(self):
        """Xem trước hiệu ứng làm mờ"""
        if not self.blur_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn làm mờ!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với kernel size đã chọn
        kernel_size = self.blur_size.value() * 2 + 1  # Đảm bảo kernel size là số lẻ
        img_blurred = apply_blur(img.copy(), kernel_size)
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_blurred)

    def preview_hue(self):
        """Xem trước hiệu ứng điều chỉnh màu sắc (hue)"""
        if not self.hue_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn điều chỉnh màu sắc!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với độ min và max
        img_min = adjust_hue(img.copy(), self.hue_min.value())
        img_max = adjust_hue(img.copy(), self.hue_max.value())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_min)
        self.preview_dialog.show_images(img, img_max)

    def preview_saturation(self):
        """Xem trước hiệu ứng điều chỉnh độ bão hòa màu"""
        if not self.sat_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn điều chỉnh độ bão hòa màu!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với độ min và max
        img_min = adjust_saturation(img.copy(), self.sat_min.value())
        img_max = adjust_saturation(img.copy(), self.sat_max.value())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_min)
        self.preview_dialog.show_images(img, img_max)

    def preview_exposure(self):
        """Xem trước hiệu ứng điều chỉnh độ phơi sáng"""
        if not self.exp_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn điều chỉnh độ phơi sáng!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với độ min và max
        img_min = adjust_exposure(img.copy(), self.exp_min.value())
        img_max = adjust_exposure(img.copy(), self.exp_max.value())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_min)
        self.preview_dialog.show_images(img, img_max)

    def preview_grayscale(self):
        """Xem trước hiệu ứng chuyển ảnh sang grayscale"""
        if not self.gray_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn chuyển ảnh sang grayscale!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với độ grayscale
        img_gray = convert_to_grayscale(img.copy())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_gray)

    def preview_gaussian_noise(self):
        """Xem trước hiệu ứng nhiễu Gaussian"""
        if not self.gaussian_noise_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn Gaussian Noise!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với phần trăm nhiễu đã chọn
        img_noisy = add_noise(img.copy(), 'gaussian', percent=self.gaussian_noise_percent.value())
        self.preview_dialog.show_images(img, img_noisy)

    def preview_salt_pepper(self):
        """Xem trước hiệu ứng nhiễu Salt & Pepper"""
        if not self.salt_pepper_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn Salt & Pepper!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với phần trăm nhiễu đã chọn
        img_noisy = add_noise(img.copy(), 'salt_pepper', percent=self.salt_pepper_percent.value())
        self.preview_dialog.show_images(img, img_noisy)

    def preview_rotate90(self):
        """Xem trước hiệu ứng xoay 90 độ"""
        if not self.rotate90_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn xoay 90 độ!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Kiểm tra xem có checkbox nào được chọn không
        if not (self.rotate90_clockwise.isChecked() or 
                self.rotate90_counter.isChecked() or 
                self.rotate90_upside.isChecked()):
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn ít nhất một hướng xoay!")
            return
            
        # Tạo ảnh preview với hướng xoay đầu tiên được chọn
        if self.rotate90_clockwise.isChecked():
            img_rotated = rotate_90(img.copy(), 'clockwise')
            self.preview_dialog.show_images(img, img_rotated)
        if self.rotate90_counter.isChecked():
            img_rotated = rotate_90(img.copy(), 'counter-clockwise')
            self.preview_dialog.show_images(img, img_rotated)
        if self.rotate90_upside.isChecked():
            img_rotated = rotate_90(img.copy(), 'upside-down')
            self.preview_dialog.show_images(img, img_rotated)

    def preview_crop(self):
        """Xem trước hiệu ứng crop"""
        if not self.crop_cb.isChecked():
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật tùy chọn crop!")
            return
            
        img = self.get_current_image()
        if img is None:
            QMessageBox.warning(self, "Cảnh báo", "Không có ảnh nào được chọn!")
            return
            
        # Tạo ảnh preview với độ crop min và max
        img_min = crop_image(img.copy(), self.crop_min.value())
        img_max = crop_image(img.copy(), self.crop_max.value())
        
        # Hiển thị preview
        self.preview_dialog.show_images(img, img_min)
        self.preview_dialog.show_images(img, img_max)

    def check_min_max(self, min_spinbox, max_spinbox):
        """Kiểm tra và cập nhật giá trị min/max"""
        min_val = min_spinbox.value()
        max_val = max_spinbox.value()
        
        if min_val > max_val:
            # Nếu min > max, đặt max = min
            max_spinbox.setValue(min_val)
            QMessageBox.warning(self, "Cảnh báo", 
                              "Giá trị Min phải nhỏ hơn hoặc bằng Max!")

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
    # Method 3
    # Chuyển numpy array thành tensor
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
    else:
        image_tensor = image
    
    # Điều chỉnh độ sáng
    adjusted = F.adjust_brightness(image_tensor, factor)
    
    # Chuyển về PIL rồi về numpy
    pil_image = T.ToPILImage()(adjusted)
    return np.array(pil_image)

    # Method 2
    # brightness_value = np.clip(factor, -99, 99)
    # offset = int(brightness_value * 255 / 99)
    # result = image.astype(np.int16) + offset
    # return np.clip(result, 0, 255).astype(np.uint8)

    # Method 1
    # return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def apply_blur(image, kernel_size):
    """Làm mờ ảnh với kernel size cho trước"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def adjust_hue(image, factor):
    """Điều chỉnh màu sắc (hue) của ảnh"""

    #Method 2
    factor = factor / 360


    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
    else:
        image_tensor = image
    
    # Điều chỉnh độ sáng
    adjusted = F.adjust_hue(image_tensor, factor)
    
    # Chuyển về PIL rồi về numpy
    pil_image = T.ToPILImage()(adjusted)
    return np.array(pil_image)

    #Method 1
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # Chuyển sang float để tính toán
    # h = hsv[:,:,0].astype(np.float32)
    # # Thực hiện phép tính
    # h = (h + factor) % 180
    # # Chuyển lại uint8
    # hsv[:,:,0] = h.astype(np.uint8)
    # return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_saturation(image, factor):
    """Điều chỉnh độ bão hòa màu của ảnh"""
    # Chuyển numpy array thành tensor
    if isinstance(image, np.ndarray):
        # Nếu là grayscale (H, W), thêm channel dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # Chuyển từ (H, W, C) sang (C, H, W) cho PyTorch
        if image.shape[2] == 3:  # RGB
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        elif image.shape[2] == 1:  # Grayscale
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        
        # Đảm bảo tensor có dtype phù hợp (float32) và giá trị trong [0, 1]
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0
        elif image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
    else:
        image_tensor = image
    
    pic = F.adjust_saturation(image_tensor, factor)  # saturation_factor = 1.0 (không thay đổi)
    pic = T.ToPILImage()(pic)
    
    return np.array(pic)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv[:,:,1] = np.clip(hsv[:,:,1] * factor, 0, 255)
    # return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_exposure(image, factor):
    """Điều chỉnh độ phơi sáng của ảnh"""
    exposure_stops = factor / 12.5
    multiplier = 2 ** exposure_stops
    
    # Áp dụng exponential scaling
    result = image.astype(np.float32) * multiplier
    
    # Clamp về [0, 255]
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)
    # return cv2.convertScaleAbs(image, alpha=1.0, beta=factor)

def convert_to_grayscale(image):
    """Chuyển ảnh sang grayscale"""
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def apply_gaussian_blur(image, kernel_size, sigma):
    """Áp dụng Gaussian blur với kernel size và sigma cho trước"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def add_noise(image, noise_type='gaussian', mean=0, sigma=25, percent=5):
    """Thêm nhiễu vào ảnh
    noise_type: 'gaussian' hoặc 'salt_pepper'
    percent: phần trăm nhiễu (1-100)
    """
    noisy = image.copy()  # Khởi tạo noisy từ ảnh gốc
    
    if noise_type.lower() == 'gaussian':
        # Tính toán sigma dựa trên phần trăm nhiễu
        adjusted_sigma = sigma * (percent / 100.0)
        noise = np.random.normal(mean, adjusted_sigma, image.shape).astype(np.uint8)
        noisy = cv2.add(noisy, noise)
    elif noise_type.lower() == 'salt_pepper':
        # Tính số lượng pixel sẽ thêm nhiễu
        num_noise = np.ceil(percent * image.size * 0.01)
        
        # Thêm nhiễu muối
        num_salt = np.ceil(num_noise * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        
        # Thêm nhiễu tiêu
        num_pepper = np.ceil(num_noise * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
    else:
        # Nếu không phải gaussian hoặc salt & pepper, trả về ảnh gốc
        return image
        
    return noisy

def rotate_90(image, direction='clockwise'):
    """Xoay ảnh 90 độ theo hướng chỉ định
    direction: 'clockwise', 'counter-clockwise', 'upside-down'
    """
    """Xoay ảnh 90 độ bằng phương pháp thủ công"""
    height, width = image.shape[:2]
    center = (width/2, height/2)
    if direction == 'clockwise':
        rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    elif direction == 'counter-clockwise':
        rotation_matrix = cv2.getRotationMatrix2D(center, -90, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    elif direction == 'upside-down':
        rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
    return image

def crop_image(image, scale):
    """Crop ảnh với tỷ lệ scale cho trước
    scale: tỷ lệ crop (0.1-1.0)
    """
    height, width = image.shape[:2]
    scale = 1 - scale / 100

    # Tính toán kích thước mới
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Tính toán vị trí bắt đầu crop
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    
    # Thực hiện crop
    cropped = image[start_y:start_y + new_height, start_x:start_x + new_width]
    
    # Resize về kích thước gốc
    resized = cv2.resize(cropped, (width, height))
    
    return resized

def has_any_augmentation(params):
    """Kiểm tra xem có kỹ thuật augmentation nào được bật không"""
    return (params.get('rotate', False) or 
            params.get('rotate90', False) or
            params.get('flip', False) or 
            params.get('bright', False) or 
            params.get('blur', False) or
            params.get('hue', False) or
            params.get('sat', False) or
            params.get('exp', False) or
            params.get('gray', False) or
            params.get('gaussian_noise', False) or
            params.get('salt_pepper', False) or
            params.get('crop', False))

def augment_image(image, params):
    """Áp dụng các kỹ thuật augmentation cho ảnh"""
    # Kiểm tra xem có kỹ thuật nào được bật không
    if not has_any_augmentation(params):
        return image.copy()  # Trả về bản copy của ảnh gốc nếu không có augmentation
    
    augmented = image.copy()
    
    # Tạo danh sách các phương pháp augmentation được bật
    enabled_methods = []
    if params.get('rotate', False):
        enabled_methods.append('rotate')
    if params.get('rotate90', False):
        # Thêm các hướng xoay 90 độ được chọn
        if params.get('rotate90_clockwise', False):
            enabled_methods.append('rotate90_clockwise')
        if params.get('rotate90_counter', False):
            enabled_methods.append('rotate90_counter')
        if params.get('rotate90_upside', False):
            enabled_methods.append('rotate90_upside')
    if params.get('flip', False):
        enabled_methods.append('flip')
    if params.get('bright', False):
        enabled_methods.append('bright')
    if params.get('blur', False):
        enabled_methods.append('blur')
    if params.get('hue', False):
        enabled_methods.append('hue')
    if params.get('sat', False):
        enabled_methods.append('sat')
    if params.get('exp', False):
        enabled_methods.append('exp')
    if params.get('gray', False):
        enabled_methods.append('gray')
    if params.get('gaussian_noise', False):
        enabled_methods.append('gaussian_noise')
    if params.get('salt_pepper', False):
        enabled_methods.append('salt_pepper')
    if params.get('crop', False):
        enabled_methods.append('crop')
        
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
            
        elif method == 'rotate90_clockwise':
            augmented = rotate_90(augmented, 'clockwise')
            params['rotate90_direction'] = 'clockwise'
            
        elif method == 'rotate90_counter':
            augmented = rotate_90(augmented, 'counter-clockwise')
            params['rotate90_direction'] = 'counter-clockwise'
            
        elif method == 'rotate90_upside':
            augmented = rotate_90(augmented, 'upside-down')
            params['rotate90_direction'] = 'upside-down'
            
        elif method == 'flip':
            h_flip = random.random() < 0.5 if params.get('flip_h', False) else False
            v_flip = random.random() < 0.5 if params.get('flip_v', False) else False
            while h_flip == False and v_flip == False:
                h_flip = random.random() < 0.5 if params.get('flip_h', False) else False
                v_flip = random.random() < 0.5 if params.get('flip_v', False) else False
            augmented = flip_image(augmented, h_flip, v_flip)
            params['flip_h'] = h_flip  # Lưu trạng thái flip để sử dụng cho bbox
            params['flip_v'] = v_flip
            
        elif method == 'bright':
            factor = random.uniform(params['bright_min'], params['bright_max'])
            augmented = adjust_brightness(augmented, factor)
            
        elif method == 'blur':
            kernel_size = params['blur_size'] * 2 + 1  # Đảm bảo kernel size là số lẻ
            augmented = apply_blur(augmented, kernel_size)
            
        elif method == 'hue':
            factor = random.uniform(params['hue_min'], params['hue_max'])
            augmented = adjust_hue(augmented, factor)
            
        elif method == 'sat':
            factor = random.uniform(params['sat_min'], params['sat_max'])
            augmented = adjust_saturation(augmented, factor)
            
        elif method == 'exp':
            factor = random.uniform(params['exp_min'], params['exp_max'])
            augmented = adjust_exposure(augmented, factor)
            
        elif method == 'gray':
            augmented = convert_to_grayscale(augmented)
            
        elif method == 'gaussian_noise':
            augmented = add_noise(augmented, 'gaussian', percent=params.get('gaussian_noise_percent', 5))
            
        elif method == 'salt_pepper':
            augmented = add_noise(augmented, 'salt_pepper', percent=params.get('salt_pepper_percent', 5))
        
        elif method == 'crop':
            scale = random.uniform(params.get('crop_min', 0.8), params.get('crop_max', 1.0))
            augmented = crop_image(augmented, scale)
            params['crop_scale'] = scale
        
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
            
            # Lưu kích thước bbox gốc
            original_bbox_width = x2 - x1
            original_bbox_height = y2 - y1
            
            # Transform tâm của bbox
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            center_point = np.array([[bbox_center_x, bbox_center_y, 1]])
            transformed_center = rotation_matrix.dot(center_point.T).T[0]
            
            # Tạo bbox mới với cùng kích thước nhưng tâm đã xoay
            new_center_x = transformed_center[0]
            new_center_y = transformed_center[1]
            
            x1 = new_center_x - original_bbox_width / 2
            y1 = new_center_y - original_bbox_height / 2
            x2 = new_center_x + original_bbox_width / 2
            y2 = new_center_y + original_bbox_height / 2
            
            # Clamp về kích thước ảnh
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
        elif method == 'rotate90':
            direction = params.get('rotate90_direction', 'clockwise')
            # Lưu kích thước bbox gốc
            original_bbox_width = x2 - x1
            original_bbox_height = y2 - y1
            
            if direction == 'clockwise':
                # Xoay 90 độ theo chiều kim đồng hồ
                new_x1 = y1
                new_y1 = width - x2
                new_x2 = y2
                new_y2 = width - x1
            elif direction == 'counter-clockwise':
                # Xoay 90 độ ngược chiều kim đồng hồ
                new_x1 = height - y2
                new_y1 = x1
                new_x2 = height - y1
                new_y2 = x2
            else:  # upside-down
                # Xoay 180 độ
                new_x1 = width - x2
                new_y1 = height - y2
                new_x2 = width - x1
                new_y2 = height - y1
                
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
            
        elif method == 'flip':
            if params.get('flip_h', False):
                x1, x2 = width - x2, width - x1
            if params.get('flip_v', False):
                y1, y2 = height - y2, height - y1
            
        elif method == 'crop':
            scale = params.get('crop_scale', 1.0)
            # Tính toán kích thước mới
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Tính toán vị trí bắt đầu crop
            start_x = (width - new_width) // 2
            start_y = (height - new_height) // 2
            
            # Tính toán tỷ lệ scale
            scale_x = width / new_width
            scale_y = height / new_height
            
            # Cập nhật tọa độ bbox
            x1 = (x1 - start_x) * scale_x
            y1 = (y1 - start_y) * scale_y
            x2 = (x2 - start_x) * scale_x
            y2 = (y2 - start_y) * scale_y
            
    # Đảm bảo tọa độ nằm trong ảnh
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    return [x1, y1, x2, y2] 