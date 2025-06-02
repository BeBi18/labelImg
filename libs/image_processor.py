import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QSpinBox, QPushButton, QGroupBox,
                            QFileDialog, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import os

class ImageProcessor:
    """Class chứa các phương thức xử lý ảnh"""
    
    @staticmethod
    def resize(image, width, height):
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def gaussian_blur(image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def binarization(image, threshold):
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def canny_edge(image, min_threshold, max_threshold):
        return cv2.Canny(image, min_threshold, max_threshold)
    
    @staticmethod
    def dilation(image, kernel_size, iterations):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=iterations)
    
    @staticmethod
    def grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

class ProcessingStep:
    """Class đại diện cho một bước xử lý ảnh"""
    def __init__(self, name, processor_func, params):
        self.name = name
        self.processor_func = processor_func
        self.params = params
        self.enabled = False

class ImageProcessorWidget(QWidget):
    """Widget chứa UI cho việc xử lý ảnh"""
    
    # Signal khi ảnh được xử lý
    image_processed = pyqtSignal(QImage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.init_processors()
        self.current_image = None
        
    def init_ui(self):
        """Khởi tạo giao diện"""
        main_layout = QVBoxLayout()
        
        # Tạo scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Widget chứa nội dung
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Group cho các bước xử lý
        self.processing_group = QGroupBox("Tiền xử lý ảnh")
        self.processing_layout = QVBoxLayout()
        self.processing_group.setLayout(self.processing_layout)
        
        # Nút Preview và Apply All
        button_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.apply_all_btn = QPushButton("Apply All")
        
        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.apply_all_btn)
        
        # Kết nối signals
        self.preview_btn.clicked.connect(self.preview_processing)
        self.apply_all_btn.clicked.connect(self.apply_all)
        
        layout.addWidget(self.processing_group)
        layout.addLayout(button_layout)
        content_widget.setLayout(layout)
        
        # Thêm content widget vào scroll area
        scroll.setWidget(content_widget)
        
        # Thêm scroll area vào main layout
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self.save_dir = None
        
    def init_processors(self):
        """Khởi tạo các bước xử lý ảnh"""
        self.processors = []
        
        # Resize
        resize_params = {
            'width': QSpinBox(),
            'height': QSpinBox()
        }
        resize_params['width'].setRange(1, 10000)
        resize_params['height'].setRange(1, 10000)
        resize_params['width'].setValue(800)
        resize_params['height'].setValue(600)
        self.add_processor("Resize", ImageProcessor.resize, resize_params)
        
        # Gaussian Blur
        blur_params = {
            'kernel_size': QSpinBox()
        }
        blur_params['kernel_size'].setRange(1, 99)
        blur_params['kernel_size'].setSingleStep(2)
        blur_params['kernel_size'].setValue(3)
        self.add_processor("Gaussian Blur", ImageProcessor.gaussian_blur, blur_params)
        
        # Binarization
        binary_params = {
            'threshold': QSpinBox()
        }
        binary_params['threshold'].setRange(0, 255)
        binary_params['threshold'].setValue(127)
        self.add_processor("Binarization", ImageProcessor.binarization, binary_params)
        
        # Canny Edge
        canny_params = {
            'min_threshold': QSpinBox(),
            'max_threshold': QSpinBox()
        }
        canny_params['min_threshold'].setRange(0, 255)
        canny_params['max_threshold'].setRange(0, 255)
        canny_params['min_threshold'].setValue(100)
        canny_params['max_threshold'].setValue(200)
        self.add_processor("Canny Edge", ImageProcessor.canny_edge, canny_params)
        
        # Dilation
        dilation_params = {
            'kernel_size': QSpinBox(),
            'iterations': QSpinBox()
        }
        dilation_params['kernel_size'].setRange(1, 99)
        dilation_params['iterations'].setRange(1, 99)
        dilation_params['kernel_size'].setValue(3)
        dilation_params['iterations'].setValue(1)
        self.add_processor("Dilation", ImageProcessor.dilation, dilation_params)
        
        # Grayscale
        self.add_processor("Grayscale", ImageProcessor.grayscale, {})
        
    def add_processor(self, name, processor_func, params):
        """Thêm một bước xử lý mới"""
        processor = ProcessingStep(name, processor_func, params)
        self.processors.append(processor)
        
        # Tạo UI cho processor
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Checkbox để bật/tắt
        checkbox = QCheckBox("Enable")
        checkbox.stateChanged.connect(lambda state: self.toggle_processor(name, state))
        layout.addWidget(checkbox)
        
        # Thêm các tham số
        for param_name, widget in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(param_name))
            param_layout.addWidget(widget)
            layout.addLayout(param_layout)
            
        group.setLayout(layout)
        self.processing_layout.addWidget(group)
        
    def toggle_processor(self, name, state):
        """Bật/tắt một bước xử lý"""
        for processor in self.processors:
            if processor.name == name:
                processor.enabled = state == Qt.Checked
                break
                
    def set_image(self, image):
        """Set ảnh hiện tại"""
        if isinstance(image, QImage):
            self.current_image = self.qimage_to_cv2(image)
        else:
            self.current_image = image
            
    def qimage_to_cv2(self, qimage):
        """Chuyển đổi QImage sang định dạng OpenCV"""
        try:
            # Chuyển QImage sang numpy array
            width = qimage.width()
            height = qimage.height()
            
            # Đảm bảo QImage ở định dạng RGB32
            if qimage.format() != QImage.Format_RGB32:
                qimage = qimage.convertToFormat(QImage.Format_RGB32)
            
            ptr = qimage.bits()
            ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            
            # Chuyển đổi từ RGBA sang BGR
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        
        except Exception as e:
            print(f"Lỗi khi chuyển đổi ảnh: {str(e)}")
            return None
        
    def cv2_to_qimage(self, cv_img):
        """Chuyển đổi ảnh OpenCV sang QImage"""
        height, width = cv_img.shape[:2]
        if len(cv_img.shape) == 2:  # Grayscale
            bytes_per_line = width
            return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color
            bytes_per_line = 3 * width
            return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
    def process_image(self, image):
        """Xử lý ảnh với các bước đã chọn"""
        if image is None:
            return None
            
        result = image.copy()
        
        for processor in self.processors:
            if processor.enabled:
                if processor.name == "Resize":
                    result = processor.processor_func(
                        result,
                        processor.params['width'].value(),
                        processor.params['height'].value()
                    )
                elif processor.name == "Gaussian Blur":
                    result = processor.processor_func(
                        result,
                        processor.params['kernel_size'].value()
                    )
                elif processor.name == "Binarization":
                    result = processor.processor_func(
                        result,
                        processor.params['threshold'].value()
                    )
                elif processor.name == "Canny Edge":
                    result = processor.processor_func(
                        result,
                        processor.params['min_threshold'].value(),
                        processor.params['max_threshold'].value()
                    )
                elif processor.name == "Dilation":
                    result = processor.processor_func(
                        result,
                        processor.params['kernel_size'].value(),
                        processor.params['iterations'].value()
                    )
                elif processor.name == "Grayscale":
                    result = processor.processor_func(result)
                    
        return result
        
    def preview_processing(self):
        """Xem trước kết quả xử lý"""
        if self.current_image is None:
            return
            
        processed = self.process_image(self.current_image)
        if processed is not None:
            qimage = self.cv2_to_qimage(processed)
            self.image_processed.emit(qimage)
            
    def apply_all(self):
        """Áp dụng xử lý cho tất cả ảnh trong thư mục"""
        # if not self.save_dir:
        #     self.select_save_dir()
        #     if not self.save_dir:
        #         return
                
        # Lấy danh sách ảnh từ parent window
        parent = self.parent()
        while parent and not hasattr(parent, 'm_img_list'):
            parent = parent.parent()
            
        if not parent or not parent.m_img_list:
            return
            
        # Chọn thư mục lưu
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục lưu ảnh đã xử lý",
            "",
            QFileDialog.ShowDirsOnly
        )
        if not dir_path:
            return
            
        self.save_dir = dir_path
            
        # Đếm số ảnh xử lý thành công
        processed_count = 0
        error_count = 0
            
        # Xử lý từng ảnh
        for img_path in parent.m_img_list:
            try:
                # Đọc ảnh
                image = cv2.imread(img_path)
                if image is None:
                    error_count += 1
                    continue
                    
                # Xử lý ảnh
                processed = self.process_image(image)
                if processed is None:
                    error_count += 1
                    continue
                    
                # Lưu ảnh đã xử lý
                filename = os.path.basename(img_path)
                save_path = os.path.join(self.save_dir, filename)
                
                # Đảm bảo thư mục tồn tại
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Lưu ảnh
                success = cv2.imwrite(save_path, processed)
                if success:
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
                error_count += 1
                continue
            
        # Thông báo kết quả
        message = f"Đã xử lý thành công: {processed_count} ảnh\n"
        if error_count > 0:
            message += f"Lỗi: {error_count} ảnh"
            
        QMessageBox.information(self, "Hoàn thành", message) 