from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QScrollArea
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QPoint
import cv2, os
import numpy as np

class ImageWidget(QLabel):
    def __init__(self, image):
        super().__init__()

        resized_image = self.load_and_resize_image(image)
        pixmap = self.pixmap_from_cv_image(resized_image)

        self.original_pixmap = pixmap
        self.setPixmap(self.original_pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scale_factor = 1.0

    def wheelEvent(self, event):
        mouse_pos = event.position().toPoint()
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in(mouse_pos)
        else:
            self.zoom_out(mouse_pos)

    def zoom_in(self, mouse_pos):
        self.scale_factor *= 1.1
        self.update_image(mouse_pos)

    def zoom_out(self, mouse_pos):
        self.scale_factor *= 0.9
        self.update_image(mouse_pos)

    def update_image(self, mouse_pos):
        size = self.original_pixmap.size()
        new_size = size * self.scale_factor
        scaled_pixmap = self.original_pixmap.scaled(new_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

        # Center the image around the mouse position
        self.recenter_image(mouse_pos)

    def recenter_image(self, mouse_pos):
        # Get the current size of the QLabel
        label_size = self.size()
        label_center = QPoint(int(label_size.width()/2), int(label_size.height()/2))

        # Calculate the offset to center the zoom effect around the mouse position
        offset_x = int(mouse_pos.x() - label_center.x())
        offset_y = int(mouse_pos.y() - label_center.y())

        # Adjust the scroll area to center the zoom effect
        scroll_area = self.parent().parent()
        scroll_bar_h = scroll_area.horizontalScrollBar()
        scroll_bar_v = scroll_area.verticalScrollBar()

        scroll_bar_h.setValue(scroll_bar_h.value() + offset_x)
        scroll_bar_v.setValue(scroll_bar_v.value() + offset_y)
    
    def load_and_resize_image(self, image, max_height=500, max_width=700, min_height=400, min_width=600):
        height = image.shape[0]
        width = image.shape[1]

        # Determine scaling factor for downscaling
        if height > max_height or width > max_width:
            if height > width:
                scaling_factor = max_height / height
            else:
                scaling_factor = max_width / width
            
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)
            image = cv2.resize(image, (new_width, new_height))
        
        # Determine scaling factor for upscaling
        elif height < min_height or width < min_width:
            if height < min_height and width < min_width:
                # Choose the scaling factor that keeps aspect ratio and meets both minimum dimensions
                scaling_factor_height = min_height / height
                scaling_factor_width = min_width / width
                scaling_factor = max(scaling_factor_height, scaling_factor_width)
            elif height < min_height:
                scaling_factor = min_height / height
            else:
                scaling_factor = min_width / width
            
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)
            image = cv2.resize(image, (new_width, new_height))
        
        return image

    def pixmap_from_cv_image(self, cv_image):   # check theory
        if len(cv_image.shape) == 2:
            cv_image = np.expand_dims(cv_image, axis=-1)

        height, width, num_channel = cv_image.shape
        
        if num_channel == 1:
            bytesPerLine = width
            qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
        else:
            bytesPerLine = 3 * width
            qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
        return QPixmap(qImg)  