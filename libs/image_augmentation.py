import os
import random
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton,
                            QFileDialog, QMessageBox, QGroupBox, QDialog,
                            QFrame, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2


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
        self.bright_min.setRange(0.5, 2.0)
        self.bright_min.setValue(0.8)
        self.bright_min.setDecimals(1)
        self.bright_min.setSingleStep(0.1)
        self.bright_min.setFixedWidth(SPINBOX_WIDTH)
        self.bright_min.valueChanged.connect(lambda: self.check_min_max(self.bright_min, self.bright_max))
        
        bright_max_label = QLabel("Max:")
        bright_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.bright_max = QDoubleSpinBox()
        self.bright_max.setRange(0.5, 2.0)
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
        
        blur_min_label = QLabel("Min:")
        blur_min_label.setFixedWidth(LABEL_WIDTH)
        
        self.blur_min = QSpinBox()
        self.blur_min.setRange(1, 10)
        self.blur_min.setValue(1)
        self.blur_min.setFixedWidth(SPINBOX_WIDTH)
        self.blur_min.valueChanged.connect(lambda: self.check_min_max(self.blur_min, self.blur_max))
        
        blur_max_label = QLabel("Max:")
        blur_max_label.setFixedWidth(LABEL_WIDTH)
        
        self.blur_max = QSpinBox()
        self.blur_max.setRange(1, 10)
        self.blur_max.setValue(3)
        self.blur_max.setFixedWidth(SPINBOX_WIDTH)
        self.blur_max.valueChanged.connect(lambda: self.check_min_max(self.blur_min, self.blur_max))
        
        self.preview_blur_btn = QPushButton("Preview")
        self.preview_blur_btn.setFixedWidth(BUTTON_WIDTH)
        
        blur_layout.addWidget(self.blur_cb)
        blur_layout.addWidget(blur_min_label)
        blur_layout.addWidget(self.blur_min)
        blur_layout.addWidget(blur_max_label)
        blur_layout.addWidget(self.blur_max)
        blur_layout.addStretch()
        blur_layout.addWidget(self.preview_blur_btn)
        
        aug_layout.addWidget(blur_frame)
        
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
            
        # Tạo ảnh preview với độ mờ min và max
        kernel_min = self.blur_min.value() * 2 + 1
        kernel_max = self.blur_max.value() * 2 + 1
        img_min = apply_blur(img.copy(), kernel_min)
        img_max = apply_blur(img.copy(), kernel_max)
        
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