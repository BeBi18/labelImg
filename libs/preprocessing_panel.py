from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import os

class PreprocessingStep:
    def __init__(self, name, enabled=False, params=None):
        self.name = name
        self.enabled = enabled
        self.params = params or {}
        
    def apply(self, image):
        if not self.enabled:
            return image
            
        if self.name == "Resize":
            width = self.params.get('width', image.shape[1])
            height = self.params.get('height', image.shape[0])
            return cv2.resize(image, (width, height))
            
        elif self.name == "Gaussian Blur":
            kernel_size = self.params.get('kernel_size', 3)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif self.name == "Binarization":
            threshold = self.params.get('threshold', 127)
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            return binary
            
        elif self.name == "Canny Edge":
            min_thresh = self.params.get('min_threshold', 100)
            max_thresh = self.params.get('max_threshold', 200)
            return cv2.Canny(image, min_thresh, max_thresh)
            
        elif self.name == "Dilation":
            kernel_size = self.params.get('kernel_size', 3)
            iterations = self.params.get('iterations', 1)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.dilate(image, kernel, iterations=iterations)
            
        elif self.name == "Grayscale":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
            
        return image

class PreprocessingPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Preprocessing", parent)
        self.parent = parent
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        self.setWidget(self.main_widget)
        
        # Initialize preprocessing steps
        self.steps = [
            PreprocessingStep("Resize", params={'width': 800, 'height': 600}),
            PreprocessingStep("Gaussian Blur", params={'kernel_size': 3}),
            PreprocessingStep("Binarization", params={'threshold': 127}),
            PreprocessingStep("Canny Edge", params={'min_threshold': 100, 'max_threshold': 200}),
            PreprocessingStep("Dilation", params={'kernel_size': 3, 'iterations': 1}),
            PreprocessingStep("Grayscale")
        ]
        
        # Create UI elements for each step
        self.step_widgets = {}
        for step in self.steps:
            self.add_step_widget(step)
            
        # Add preview label
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.preview_label)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_processing)
        button_layout.addWidget(self.preview_button)
        
        self.apply_all_button = QPushButton("Apply All")
        self.apply_all_button.clicked.connect(self.apply_all)
        button_layout.addWidget(self.apply_all_button)
        
        self.layout.addLayout(button_layout)
        
    def add_step_widget(self, step):
        group = QGroupBox(step.name)
        layout = QVBoxLayout()
        
        # Enable checkbox
        enable_cb = QCheckBox("Enable")
        enable_cb.setChecked(step.enabled)
        enable_cb.stateChanged.connect(lambda state: self.update_step_state(step, state))
        layout.addWidget(enable_cb)
        
        # Add parameter inputs based on step type
        if step.name == "Resize":
            width_spin = QSpinBox()
            width_spin.setRange(1, 9999)
            width_spin.setValue(step.params['width'])
            width_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'width', v))
            layout.addWidget(QLabel("Width:"))
            layout.addWidget(width_spin)
            
            height_spin = QSpinBox()
            height_spin.setRange(1, 9999)
            height_spin.setValue(step.params['height'])
            height_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'height', v))
            layout.addWidget(QLabel("Height:"))
            layout.addWidget(height_spin)
            
        elif step.name == "Gaussian Blur":
            kernel_spin = QSpinBox()
            kernel_spin.setRange(1, 99)
            kernel_spin.setValue(step.params['kernel_size'])
            kernel_spin.setSingleStep(2)
            kernel_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'kernel_size', v))
            layout.addWidget(QLabel("Kernel Size:"))
            layout.addWidget(kernel_spin)
            
        elif step.name == "Binarization":
            threshold_spin = QSpinBox()
            threshold_spin.setRange(0, 255)
            threshold_spin.setValue(step.params['threshold'])
            threshold_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'threshold', v))
            layout.addWidget(QLabel("Threshold:"))
            layout.addWidget(threshold_spin)
            
        elif step.name == "Canny Edge":
            min_thresh_spin = QSpinBox()
            min_thresh_spin.setRange(0, 255)
            min_thresh_spin.setValue(step.params['min_threshold'])
            min_thresh_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'min_threshold', v))
            layout.addWidget(QLabel("Min Threshold:"))
            layout.addWidget(min_thresh_spin)
            
            max_thresh_spin = QSpinBox()
            max_thresh_spin.setRange(0, 255)
            max_thresh_spin.setValue(step.params['max_threshold'])
            max_thresh_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'max_threshold', v))
            layout.addWidget(QLabel("Max Threshold:"))
            layout.addWidget(max_thresh_spin)
            
        elif step.name == "Dilation":
            kernel_spin = QSpinBox()
            kernel_spin.setRange(1, 99)
            kernel_spin.setValue(step.params['kernel_size'])
            kernel_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'kernel_size', v))
            layout.addWidget(QLabel("Kernel Size:"))
            layout.addWidget(kernel_spin)
            
            iter_spin = QSpinBox()
            iter_spin.setRange(1, 99)
            iter_spin.setValue(step.params['iterations'])
            iter_spin.valueChanged.connect(lambda v: self.update_step_param(step, 'iterations', v))
            layout.addWidget(QLabel("Iterations:"))
            layout.addWidget(iter_spin)
            
        group.setLayout(layout)
        self.layout.addWidget(group)
        self.step_widgets[step.name] = group
        
    def update_step_state(self, step, state):
        step.enabled = bool(state)
        
    def update_step_param(self, step, param_name, value):
        step.params[param_name] = value
        
    def preview_processing(self):
        if not self.parent or not self.parent.image:
            return
            
        # Convert QImage to numpy array
        image = self.parent.image
        if isinstance(image, QImage):
            image = self.qimage_to_numpy(image)
            
        # Apply all enabled steps
        processed = image.copy()
        for step in self.steps:
            if step.enabled:
                processed = step.apply(processed)
                
        # Convert back to QImage and display
        preview = self.numpy_to_qimage(processed)
        self.preview_label.setPixmap(QPixmap.fromImage(preview))
        
    def apply_all(self):
        if not self.parent or not self.parent.m_img_list:
            return
            
        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
            
        # Process all images
        for img_path in self.parent.m_img_list:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Apply all enabled steps
            processed = image.copy()
            for step in self.steps:
                if step.enabled:
                    processed = step.apply(processed)
                    
            # Save processed image
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(save_dir, f"{name}_prep{ext}")
            cv2.imwrite(save_path, processed)
            
    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        
    def numpy_to_qimage(self, arr):
        height, width = arr.shape[:2]
        bytes_per_line = 3 * width
        return QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888) 