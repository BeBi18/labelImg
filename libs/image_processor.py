import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QSpinBox, QPushButton, QGroupBox,
                            QFileDialog, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPixmap
import os
from libs.utils import generate_color_by_text
from libs.hashableQListWidgetItem import HashableQListWidgetItem

class ImageProcessor:
    """Các phương thức xử lý ảnh"""
    
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
    def erosion(image, kernel_size, iterations):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=iterations)
    
    @staticmethod
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

class ProcessingStep:
    """Một bước xử lý ảnh"""
    def __init__(self, name, processor_func, params):
        self.name = name
        self.processor_func = processor_func
        self.params = params
        self.enabled = False

class ImageProcessorWidget(QWidget):
    """Widget UI cho việc xử lý ảnh"""
    
    image_processed = pyqtSignal(QImage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.original_bboxes = []
        self.current_bboxes = []
        self.original_shape = None
        self.save_dir = None
        
        self.init_ui()
        self.init_processors()
        
    def init_ui(self):
        """Khởi tạo giao diện"""
        main_layout = QVBoxLayout()
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Group xử lý
        self.processing_group = QGroupBox("Tiền xử lý ảnh")
        self.processing_layout = QVBoxLayout()
        self.processing_group.setLayout(self.processing_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.apply_all_btn = QPushButton("Apply All")
        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.apply_all_btn)
        
        self.preview_btn.clicked.connect(self.preview_processing)
        self.apply_all_btn.clicked.connect(self.apply_all)
        
        layout.addWidget(self.processing_group)
        layout.addLayout(button_layout)
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
    def init_processors(self):
        """Khởi tạo các bước xử lý"""
        self.processors = []
        
        # Resize
        resize_params = {'width': QSpinBox(), 'height': QSpinBox()}
        resize_params['width'].setRange(1, 10000)
        resize_params['height'].setRange(1, 10000)
        resize_params['width'].setValue(800)
        resize_params['height'].setValue(600)
        self.add_processor("Resize", ImageProcessor.resize, resize_params)
        
        # Gaussian Blur
        blur_params = {'kernel_size': QSpinBox()}
        blur_params['kernel_size'].setRange(1, 99)
        blur_params['kernel_size'].setSingleStep(2)
        blur_params['kernel_size'].setValue(3)
        self.add_processor("Gaussian Blur", ImageProcessor.gaussian_blur, blur_params)
        
        # Binarization
        binary_params = {'threshold': QSpinBox()}
        binary_params['threshold'].setRange(0, 255)
        binary_params['threshold'].setValue(127)
        self.add_processor("Binarization", ImageProcessor.binarization, binary_params)
        
        # Canny Edge
        canny_params = {'min_threshold': QSpinBox(), 'max_threshold': QSpinBox()}
        canny_params['min_threshold'].setRange(0, 255)
        canny_params['max_threshold'].setRange(0, 255)
        canny_params['min_threshold'].setValue(100)
        canny_params['max_threshold'].setValue(200)
        self.add_processor("Canny Edge", ImageProcessor.canny_edge, canny_params)
        
        # Dilation
        dilation_params = {'kernel_size': QSpinBox(), 'iterations': QSpinBox()}
        dilation_params['kernel_size'].setRange(1, 99)
        dilation_params['iterations'].setRange(1, 99)
        dilation_params['kernel_size'].setValue(3)
        dilation_params['iterations'].setValue(1)
        self.add_processor("Dilation", ImageProcessor.dilation, dilation_params)
        
        # Erosion
        erosion_params = {'kernel_size': QSpinBox(), 'iterations': QSpinBox()}
        erosion_params['kernel_size'].setRange(1, 99)
        erosion_params['iterations'].setRange(1, 99)
        erosion_params['kernel_size'].setValue(3)
        erosion_params['iterations'].setValue(1)
        self.add_processor("Erosion", ImageProcessor.erosion, erosion_params)
        
        # Grayscale
        self.add_processor("Grayscale", ImageProcessor.grayscale, {})
        
    def add_processor(self, name, processor_func, params):
        """Thêm processor và tạo UI"""
        processor = ProcessingStep(name, processor_func, params)
        self.processors.append(processor)
        
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        checkbox = QCheckBox("Enable")
        checkbox.stateChanged.connect(lambda state: self.toggle_processor(name, state))
        layout.addWidget(checkbox)
        
        for param_name, widget in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(param_name))
            param_layout.addWidget(widget)
            layout.addLayout(param_layout)
            
        group.setLayout(layout)
        self.processing_layout.addWidget(group)
        
    def toggle_processor(self, name, state):
        """Bật/tắt processor"""
        for processor in self.processors:
            if processor.name == name:
                processor.enabled = state == Qt.Checked
                
                if name == "Resize" and not processor.enabled:
                    self.current_bboxes = self.original_bboxes.copy()
                    # self.update_canvas_bboxes()
                break
                
    def update_canvas_bboxes(self):
        """Cập nhật bbox trên canvas"""
        parent = self.get_main_parent()
        if not parent or not hasattr(parent, 'canvas'):
            return
            
        try:
            from libs.canvas import Shape
        except ImportError:
            try:
                from canvas import Shape
            except ImportError:
                return
                
        # Clear existing shapes
        for shape in parent.canvas.shapes:
            if shape in parent.shapes_to_items:
                item = parent.shapes_to_items[shape]
                parent.label_list.takeItem(parent.label_list.row(item))
                del parent.items_to_shapes[item]
                del parent.shapes_to_items[shape]
        
        parent.canvas.shapes = []
        
        # Add new shapes
        for bbox in self.current_bboxes:
            points = [
                QPointF(bbox[0], bbox[1]),
                QPointF(bbox[2], bbox[1]),
                QPointF(bbox[2], bbox[3]),
                QPointF(bbox[0], bbox[3])
            ]
            
            shape = Shape(label='object')
            shape.points = points
            shape.close()
            parent.canvas.shapes.append(shape)
            
            item = HashableQListWidgetItem(shape.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setBackground(generate_color_by_text(shape.label))
            
            parent.shapes_to_items[shape] = item
            parent.items_to_shapes[item] = shape
            parent.label_list.addItem(item)
            
        parent.canvas.update()
                
    def set_image(self, image):
        """Set ảnh hiện tại"""
        self.current_image = self.qimage_to_cv2(image) if isinstance(image, QImage) else image
        self.original_shape = self.current_image.shape[:2]
            
    def qimage_to_cv2(self, qimage):
        """QImage -> OpenCV"""
        try:
            width, height = qimage.width(), qimage.height()
            
            if qimage.format() != QImage.Format_RGB32:
                qimage = qimage.convertToFormat(QImage.Format_RGB32)
            
            ptr = qimage.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            print(f"Lỗi chuyển đổi ảnh: {e}")
            return None
        
    def cv2_to_qimage(self, cv_img):
        """OpenCV -> QImage"""
        height, width = cv_img.shape[:2]
        if len(cv_img.shape) == 2:
            return QImage(cv_img.data, width, height, width, QImage.Format_Grayscale8)
        else:
            return QImage(cv_img.data, width, height, 3 * width, QImage.Format_RGB888)
        
    def set_bboxes(self, bboxes):
        """Set bbox"""
        self.original_bboxes = bboxes.copy() if bboxes else []
        self.current_bboxes = bboxes.copy() if bboxes else []
        
    def get_current_bboxes(self):
        return self.current_bboxes
        
    def extract_point_coordinates(self, point):
        """Trích xuất tọa độ từ point"""
        if isinstance(point, QPointF):
            return (point.x(), point.y())
        elif isinstance(point, tuple) and len(point) == 2:
            return point
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return (point.x(), point.y())
        else:
            try:
                return (float(point[0]), float(point[1]))
            except:
                return (0, 0)
    
    def get_main_parent(self):
        """Lấy parent window chính"""
        parent = self.parent()
        while parent and not hasattr(parent, 'canvas'):
            parent = parent.parent()
        return parent
        
    def process_image(self, image, is_preview=False):
        """Xử lý ảnh với các bước đã chọn"""
        if image is None:
            return None
            
        result = image.copy()
        
        if is_preview:
            self.current_bboxes = self.original_bboxes.copy()
        
        for processor in self.processors:
            if not processor.enabled:
                continue
                
            if processor.name == "Resize":
                new_width = processor.params['width'].value()
                new_height = processor.params['height'].value()
                new_shape = (new_height, new_width)
                
                # Update bbox
                self.current_bboxes = [
                    update_bbox_for_resize(bbox, self.original_shape, new_shape)
                    for bbox in self.current_bboxes
                ]
                
                result = processor.processor_func(result, new_width, new_height)
                
            elif processor.name == "Gaussian Blur":
                result = processor.processor_func(result, processor.params['kernel_size'].value())
                
            elif processor.name == "Binarization":
                result = processor.processor_func(result, processor.params['threshold'].value())
                
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
                
            elif processor.name == "Erosion":
                result = processor.processor_func(
                    result,
                    processor.params['kernel_size'].value(),
                    processor.params['iterations'].value()
                )
                
            elif processor.name == "Grayscale":
                result = processor.processor_func(result)
                    
        return result
        
    def preview_processing(self):
        """Preview xử lý"""
        if self.current_image is None:
            return
            
        parent = self.get_main_parent()
        
        # Lưu bbox từ canvas nếu chưa có
        if parent and hasattr(parent, 'canvas') and not self.original_bboxes:
            self.original_bboxes = []
            for shape in parent.canvas.shapes:
                if len(shape.points) >= 4:
                    try:
                        point1 = self.extract_point_coordinates(shape.points[0])
                        point3 = self.extract_point_coordinates(shape.points[2])
                        x1, y1 = point1
                        x2, y2 = point3
                        self.original_bboxes.append([x1, y1, x2, y2])
                    except Exception as e:
                        print(f"Lỗi trích xuất bbox: {e}")
                        continue
                        
        self.current_bboxes = self.original_bboxes.copy()
        
        # Xử lý ảnh
        processed = self.process_image(self.current_image, is_preview=True)
        if processed is not None:
            qimage = self.cv2_to_qimage(processed)
            self.image_processed.emit(qimage)
            self.update_canvas_bboxes()
            
    def apply_all(self):
        """Áp dụng cho tất cả ảnh"""
        parent = self.get_main_parent()
        if not parent or not hasattr(parent, 'm_img_list') or not parent.m_img_list:
            return
            
        # Chọn thư mục lưu
        dir_path = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu ảnh đã xử lý", "", QFileDialog.ShowDirsOnly
        )
        if not dir_path:
            return
            
        processed_count = error_count = 0
            
        # Xử lý từng ảnh
        for img_path in parent.m_img_list:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    error_count += 1
                    continue
                    
                processed = self.process_image(image)
                if processed is None:
                    error_count += 1
                    continue
                    
                filename = os.path.basename(img_path)
                save_path = os.path.join(dir_path, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                if cv2.imwrite(save_path, processed):
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"Lỗi xử lý ảnh {img_path}: {e}")
                error_count += 1
            
        # Thông báo kết quả
        message = f"Đã xử lý thành công: {processed_count} ảnh"
        if error_count > 0:
            message += f"\nLỗi: {error_count} ảnh"
            
        QMessageBox.information(self, "Hoàn thành", message)

def update_bbox_for_resize(bbox, original_shape, new_shape):
    """Cập nhật bbox sau resize"""
    orig_height, orig_width = original_shape[:2]
    new_height, new_width = new_shape[:2]
    
    x1, y1, x2, y2 = bbox
    
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    x1 = max(0, min(x1 * scale_x, new_width))
    y1 = max(0, min(y1 * scale_y, new_height))
    x2 = max(0, min(x2 * scale_x, new_width))
    y2 = max(0, min(y2 * scale_y, new_height))
    
    return [x1, y1, x2, y2]