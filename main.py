import sys
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QComboBox, QVBoxLayout, QWidget, QFileDialog, QLabel, QErrorMessage)
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
import cv2
import numpy as np
import math
from queue import Queue
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd

# file imports
from Image_Resize import nearest_neighbor_im, bilinear_im
import Image_Math as IM_MAT
import Image_Labeling as IM_LABEL

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def hasHeightForWidth(self):
        return self.pixmap() is not None

    def heightForWidth(self, w):
        if self.pixmap():
            try:
                return int(w * (self.pixmap().height() / self.pixmap().width()))
            except ZeroDivisionError:
                return 0


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


def resize_image(image_data, max_img_width, max_img_height):
    scale_percent = min(max_img_width / image_data.shape[1], max_img_height / image_data.shape[0])
    width = int(image_data.shape[1] * scale_percent)
    height = int(image_data.shape[0] * scale_percent)
    newSize = (width, height)
    image_resized = cv2.resize(image_data, newSize, None, None, None, cv2.INTER_AREA)
    return image_resized


def pixmap_from_cv_image(cv_image):
    height, width, _ = cv_image.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
    return QPixmap(qImg)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Simple Image Processor")

        main_layout = QVBoxLayout()
        top_bar_layout = QHBoxLayout()
        mid_bar_layout = QHBoxLayout()
        mid2_bar_layout = QHBoxLayout()
        mid3_bar_layout = QHBoxLayout()
        image_bar_layout = QHBoxLayout()
        hist_layout = QHBoxLayout()
        self.source_filename = None
        self.source_image_data = None
        self.additional_filename = None
        self.additional_image_data = None
        self.result_image_data = None
        self.result_ccl_labels = None
        self.threshold = 120 # Threshold set to a little darker than 255/2
        self.max_img_height = 400
        self.max_img_width = 600
        self.histogram_df = pd.DataFrame()
        self.histogram_canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.histogram_canvas2 = MplCanvas(self, width=5, height=4, dpi=100)

        select_image_button = QPushButton('Select Image')
        gamma_process_image_button = QPushButton('Process Gamma')
        log_process_image_button = QPushButton('Process Log')
        add_image_button = QPushButton('Add')
        subtract_image_button = QPushButton('Subtract')
        product_image_button = QPushButton('Product')
        negative_image_button = QPushButton('Negative')
        # ALEX INSERT BEGIN
        red_mask_image_button = QPushButton('Red Mask')
        red_mask_image_button.clicked.connect(self.red_mask)
        # ALEX INSERT END
        ccl_button = QPushButton('CCL')
        self.interpolation_combo_box = QComboBox()
        self.stacking_combo_box = QComboBox()
        self.foreground_select = QComboBox()
        self.to_result = QCheckBox('Apply to result')
        self.show_histograms = QCheckBox('Show Histograms')
        histogram_equal = QPushButton("Histogram Equalization")

        # Button Connections #
        select_image_button.clicked.connect(self.choose_source_image)
        gamma_process_image_button.clicked.connect(self.gamma_image)
        log_process_image_button.clicked.connect(self.log_image)

        add_image_button.clicked.connect(self.add_images)
        subtract_image_button.clicked.connect(self.subtract_images)
        product_image_button.clicked.connect(self.multiply_images)
        negative_image_button.clicked.connect(self.negate_image)
        ccl_button.clicked.connect(self.ccl_image)

        self.interpolation_combo_box.addItems(["Nearest Neighbor", "Bilinear"])
        self.stacking_combo_box.addItems(["Averaging", "Max and Min", "Median", "LRGB",
                                            "Narrowband", "Sigma"])
        self.foreground_select.addItems(["Foreground is lighter than", "Foreground is darker than"])
        self.gamma_c_select = QDoubleSpinBox()
        self.gamma_select = QDoubleSpinBox()
        self.log_c_select = QDoubleSpinBox()
        self.log_select = QDoubleSpinBox()
        self.threshold_select = QSpinBox()
        for start_val, prefix, spinbox in zip([99.99], ['Gamma C'],
                                              [self.gamma_c_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0.01)
            spinbox.setMaximum(99.99)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(210)
            # setting decimal precision
            spinbox.setDecimals(2)

        for start_val, prefix, spinbox in zip([0.01], ['Gamma'],
                                              [self.gamma_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0.01)
            spinbox.setMaximum(3.87)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(180)
            # setting decimal precision
            spinbox.setDecimals(2)

        for start_val, prefix, spinbox in zip([99.99], ['Log C'],
                                              [self.log_c_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0.01)
            spinbox.setMaximum(99.99)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(170)
            # setting decimal precision
            spinbox.setDecimals(2)

        for start_val, prefix, spinbox in zip([0.01], ['Log Base'],
                                              [self.log_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0.01)
            spinbox.setMaximum(10.00)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(170)
            # setting decimal precision
            spinbox.setDecimals(2)

        for start_val, prefix, spinbox in zip([0.01], ['Log Base'],
                                              [self.log_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0.01)
            spinbox.setMaximum(10.00)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(130)
            # setting decimal precision
            spinbox.setDecimals(2)

        for start_val, prefix, spinbox in zip([120], ['Threshold'],
                                              [self.threshold_select]):
            spinbox.setPrefix(f'{prefix}:')
            spinbox.setMinimum(0)
            spinbox.setMaximum(255)
            spinbox.setValue(start_val)
            spinbox.setFixedWidth(170)

        top_bar_layout.addWidget(select_image_button)
        top_bar_layout.addWidget(self.gamma_c_select)
        top_bar_layout.addWidget(self.gamma_select)
        top_bar_layout.addWidget(self.log_c_select)
        top_bar_layout.addWidget(self.log_select)
        top_bar_layout.addWidget(gamma_process_image_button)
        top_bar_layout.addWidget(log_process_image_button)

        mid_bar_layout.addWidget(add_image_button)
        mid_bar_layout.addWidget(subtract_image_button)
        mid_bar_layout.addWidget(product_image_button)
        mid_bar_layout.addWidget(negative_image_button)
        # ALEX CODE INSERT START
        mid_bar_layout.addWidget(red_mask_image_button)
        # ALEX CODE INSERT END
        mid2_bar_layout.addWidget(self.stacking_combo_box)
        mid2_bar_layout.addWidget(self.interpolation_combo_box)
        mid2_bar_layout.addWidget(ccl_button)

        mid3_bar_layout.addWidget(self.to_result)
        mid3_bar_layout.addWidget(self.foreground_select)
        mid3_bar_layout.addWidget(self.threshold_select)

        self.source_image = ImageWidget()
        self.result_image = ImageWidget()
        self.source_image.setMaximumSize(self.max_img_width, self.max_img_height)
        self.result_image.setMaximumSize(self.max_img_width, self.max_img_height)

        source_image_layout = QVBoxLayout()
        source_image_layout.addWidget(QLabel("Source image:"))
        source_image_layout.addWidget(self.source_image)

        result_image_layout = QVBoxLayout()
        result_image_layout.addWidget(QLabel("Result image:"))
        result_image_layout.addWidget(self.result_image)

        image_bar_layout.addLayout(source_image_layout)
        image_bar_layout.addLayout(result_image_layout)

        bottom_bar_layout = QHBoxLayout()
        self.save_button = QPushButton('Save as file')
        self.save_button.clicked.connect(self.save_as_file)
        self.save_button.setFixedWidth(300)
        bottom_bar_layout.addWidget(self.save_button)

        bottom_bar2_layout = QHBoxLayout()
        bottom_bar2_layout.addWidget(self.show_histograms)
        bottom_bar2_layout.addWidget(histogram_equal)

        source_hist_layout = QVBoxLayout()
        source_hist_layout.addWidget(QLabel("Source Histogram:"))
        # bottom_bar3_layout.addWidget()

        result_hist_layout = QVBoxLayout()
        result_hist_layout.addWidget(QLabel("Result Histogram:"))
        # bottom_bar3_layout.addWidget()

        hist_layout.addLayout(source_hist_layout)
        hist_layout.addLayout(result_hist_layout)

        main_layout.addLayout(top_bar_layout)
        main_layout.addLayout(mid_bar_layout)
        main_layout.addLayout(mid2_bar_layout)
        main_layout.addLayout(mid3_bar_layout)
        main_layout.addLayout(image_bar_layout)
        main_layout.addLayout(bottom_bar_layout)
        main_layout.addLayout(bottom_bar2_layout)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    

    def resize_image(self, image_data, max_img_width, max_img_height):
        scale_percent = min(max_img_width / image_data.shape[1], max_img_height / image_data.shape[0])
        width = int(image_data.shape[1] * scale_percent)
        height = int(image_data.shape[0] * scale_percent)
        newSize = (width, height)
        if self.interpolation_combo_box.currentText() == "Nearest Neighbor":
            image_resized = nearest_neighbor_im(image_data, image_data.shape[1], image_data.shape[0])
            #image_resized = cv2.resize(image_data, newSize, None, None, None, cv2.INTER_NEAREST)
            return image_resized
        elif self.interpolation_combo_box.currentText() == "Bilinear":
            image_resized = bilinear_im(image_data, image_data.shape[1], image_data.shape[0])
            #image_resized = cv2.resize(image_data, newSize, None, None, None, cv2.INTER_LINEAR)
            return image_resized

    def resizeEvent(self, event):
        # This function is called when the window is resized
        if self.source_image_data is not None:
            source_image_resized = self.resize_image(self.source_image_data, self.max_img_width, self.max_img_height)
            self.source_image.setPixmap(pixmap_from_cv_image(source_image_resized))
        if self.result_image_data is not None:
            result_image_resized = self.resize_image(self.result_image_data, self.max_img_width, self.max_img_height)
            self.result_image.setPixmap(pixmap_from_cv_image(result_image_resized))

    def choose_source_image(self):
        self.source_filename = QFileDialog.getOpenFileName()[0]
        self.source_image_data = cv2.imread(self.source_filename)
        source_image_resized = resize_image(self.source_image_data, self.max_img_width, self.max_img_height)
        self.source_image.setPixmap(pixmap_from_cv_image(source_image_resized))

    ## Cassie Code Start ##
    def add_images(self):
        if self.source_image_data is None or self.additional_image_data is None:
            return # maybe display an error message?
        self.result_image_data = IM_MAT.im_add(self.source_image_data, self.additional_image_data)
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))

    def subtract_images(self):
        if self.source_image_data is None or self.additional_image_data is None:
            return # maybe display an error message?
        self.result_image_data = IM_MAT.im_sub(self.source_image_data, self.additional_image_data)
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))
    
    def multiply_images(self):
        if self.source_image_data is None or self.additional_image_data is None:
            return # maybe display an error message?
        self.result_image_data = IM_MAT.im_mult(self.source_image_data, self.additional_image_data)
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))
    
    def negate_image(self):
        if self.source_image_data is None:
            return # maybe display an error message?
        self.result_image_data = IM_MAT.im_negative(self.source_image_data)
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))
    
    def log_image(self):
        if self.source_image_data is None:
            return # maybe display an error message?
        self.result_image_data = IM_MAT.im_log(self.source_image_data, self.log_c_select.value(), self.log_select.value())
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))
    
    def gamma_image(self):
        if self.source_image_data is None:
            return # maybe display an error message?
        self.result_image_data = IM_MAT.im_gamma(self.source_image_data, self.gamma_c_select.value(), self.gamma_select.value())
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))
    
    def ccl_image(self):
        if self.source_image_data is None:
            return # maybe display an error message?
        self.result_ccl_labels = IM_LABEL.better_ccl(self.source_image_data)
        self.result_image_data = IM_LABEL.labels_to_image(self.result_ccl_labels) # REPLACE LATER
        self.result_image.setPixmap(pixmap_from_cv_image(self.result_image_data))
    ## Cassie Code End ##

    def save_as_file(self):
        if self.result_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('No image processed')
            error_dialog.exec()
        else:
            filename = QFileDialog.getSaveFileName(self, 'Save File')[0]
            if len(filename) > 0:
                cv2.imwrite(filename, self.result_image_data)

    def red_mask(self):
        masked_image = np.copy(self.source_image_data)
        for y in range(self.source_image_data.shape[0]):
            for x in range(self.source_image_data.shape[1]):
                red_value = self.source_image_data[y, x, 0]
                mask_value = 255 - red_value
                masked_image[y, x, 0] = mask_value
                masked_image[y, x, 1] = mask_value
                masked_image[y, x, 2] = mask_value
        self.source_image.setPixmap(pixmap_from_cv_image(masked_image))

def run_app():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()



if __name__ == '__main__':
    run_app()