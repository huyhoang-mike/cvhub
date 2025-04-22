from PyQt6.QtWidgets import (
QMainWindow, QApplication, QFileDialog, QListWidgetItem, QListWidget, 
QLabel, QMessageBox, QToolButton, QWidget, QVBoxLayout, QTextBrowser,
QPushButton, QAbstractItemView, QToolTip
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette, QIcon, QPainter
from PyQt6.QtCore import Qt, QSize
from ui.mainWindow import Ui_MainWindow
import os, sys
import cv2
import albumentations as A
from utils.gallery import gridInit, legendsInit, create_legends_instance, LegendsManager, colors, yoloInit
from utils.album import ImageWidget
from utils.path import load_images_and_labels
from transform.helper import borderFlag, interpolationFlag
from transform.augmentations import augmentationFolder
from transform.preprocessing import preprocessedFolder
from utils.pytorch import get_all_available_devices, getDevice
from models.classification import (
    classificationTrainThread, classificationEvalThread, classificationInferenceThread, 
    classificationModel, classificationInferenceThread_test
)
from models.detection import (
    yoloTrainThread, yoloEvalThread, yoloInferThread
)
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pyqtgraph as pg
from utils.graph import ConfusionMatrixCanvas
import numpy as np
from PyQt6.QtCharts import QChart, QChartView, QPieSeries
from PyQt6.QtGui import QPainter, QPen
from templates.pytorch.generate import generate_script
from pyqcodeeditor.QCodeEditor import QCodeEditor
from pyqcodeeditor.highlighters import QPythonHighlighter

class mainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setupStatus()
        self.signalControl()

        self.image_files = []
        self.current_index = 0
        self.current_index_p = 0
        self.current_inf_index = 0

        self.click = 0
        self.pipelineMode = 0
        self.inferenceMode = "dataloader"
        self.inference_image = None
        self.inference_folder = None
        self.current_inf_index_folder = 0
        self.projectFolder = None
        self.projectBrowse = None
        self.imagesFolder = None
        self.augmentedFolder = None
        self.preprocessedFolder = None
        self.trainSources = None
        self.outputsFolder = None

        self.augmentations = A.Compose([])
        self.augmentationsList = set()
        self.preprocesses = []
        self.preprocessesList = set()
        
        self.transformA = None
        self.transformP = None

        pixmap = QPixmap("C:/Users/huyhoang.nguyen@infineon.com/Desktop/DLGUI/ui/resources/generative.png") 
        icon = QIcon(pixmap)
        self.setWindowIcon(icon)

        self.setWindowTitle("Deep Learning Tools")
        self.show()

    def setupStatus(self):
        
        self.companyName = QLabel("Infineon Technologies AG")
        self.ui.statusbar.addPermanentWidget(self.companyName)

        # disable all tab
        for i in range(1, self.ui.tabWidget.count()):
            self.ui.tabWidget.tabBar().setTabTextColor(i, QColor('grey'))
            self.ui.tabWidget.setTabEnabled(i, False)

        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.augmentation_stack.setCurrentIndex(0)
        self.ui.preprocess_stack.setCurrentIndex(0)
        self.ui.train_mainwindow_tab.setCurrentIndex(0)
        self.ui.evaluate_mainwindow_tab.setCurrentIndex(0)
        self.ui.dataset_tab.setCurrentIndex(0)
        self.ui.analysis_tab.setCurrentIndex(0)

        self.ui.nn_device.addItems(get_all_available_devices())
        self.ui.eval_device.addItems(get_all_available_devices())

        self.ui.augmentation_usage.setToolTip('This <b>augmentation tab</b> is used to create more images. For example, when you have 100 images and choose x3, the final folder will contain 300 augmented images and 100 original images')
        self.ui.preprocess_usage.setToolTip('This <b>preprocessing tab</b> is used to preprocess all the images from the augmentation step')

    def create_colored_square_icon(self, color):
        pixmap = QPixmap(20, 20)  # Create a 20x20 square
        pixmap.fill(QColor(0, 0, 0, 0))  # Fill with transparent background
        
        painter = QPainter(pixmap)
        painter.setBrush(color)  # Set brush to the specified color
        painter.setPen(QColor(0, 0, 0, 0))  # No border
        painter.drawRect(0, 0, 20, 20)  # Draw the square
        painter.end()
        
        return QIcon(pixmap)

    def signalControl(self):
        # menu
        self.ui.actionClassification.triggered.connect(self.DatasetInit)
        self.ui.actionDetection.triggered.connect(self.det_dataset)
        self.ui.actionSegementation.triggered.connect(self.seg_dataset)
        self.ui.project_browse.clicked.connect(self.browse)
        self.ui.project_create.clicked.connect(self.createProject)
        
        # project
        self.ui.classification.mousePressEvent = lambda event: self.selectProject(0)
        self.ui.instance.mousePressEvent = lambda event: self.selectProject(1)
        self.ui.detection.mousePressEvent = lambda event: self.selectProject(2)
        self.ui.semantic.mousePressEvent = lambda event: self.selectProject(3)
        
        # dataset
        self.ui.upload_image_folder.clicked.connect(self.DatasetInit)
        # self.ui.upload_image_folder.clicked.connect(self.yoloDatasetInit)
        self.ui.dataset_classes_list.itemClicked.connect(self.classClicked)
        self.ui.apply_pipeline.clicked.connect(self.dropTab)

        # augmentation
        self.ui.aug_next.clicked.connect(self.show_next_image)
        self.ui.aug_back.clicked.connect(self.show_previous_image)
        self.ui.augmentation_options.currentIndexChanged.connect(self.stack_control)

        self.ui.crop_width_a.sliderReleased.connect(lambda: self.doAugmentation("Crop"))
        self.ui.crop_height_a.sliderReleased.connect(lambda: self.doAugmentation("Crop"))
        self.ui.crop_p.valueChanged.connect(lambda: self.doAugmentation("Crop"))
        self.ui.noise_std.sliderReleased.connect(lambda: self.doAugmentation("Gauss Noise"))
        self.ui.noise_mean.sliderReleased.connect(lambda: self.doAugmentation("Gauss Noise"))
        self.ui.noise_channel_check.stateChanged.connect(lambda: self.doAugmentation("Gauss Noise"))
        self.ui.noise_scale.sliderReleased.connect(lambda: self.doAugmentation("Gauss Noise"))
        self.ui.noise_p.valueChanged.connect(lambda: self.doAugmentation("Gauss Noise"))
        self.ui.rotate_limit.sliderReleased.connect(lambda: self.doAugmentation("Rotate"))
        self.ui.rotate_inter_options.currentIndexChanged.connect(lambda: self.doAugmentation("Rotate"))
        self.ui.rotate_border_options.currentIndexChanged.connect(lambda: self.doAugmentation("Rotate"))
        self.ui.rotate_p.valueChanged.connect(lambda: self.doAugmentation("Rotate"))
        self.ui.horizontal_check.stateChanged.connect(lambda: self.doAugmentation("Horizontal Flip"))
        self.ui.horizontal_p.valueChanged.connect(lambda: self.doAugmentation("Horizontal Flip"))
        self.ui.vertical_check.stateChanged.connect(lambda: self.doAugmentation("Vertical Flip"))
        self.ui.vertical_p.valueChanged.connect(lambda: self.doAugmentation("Vertical Flip"))
        self.ui.dropout_holes.sliderReleased.connect(lambda: self.doAugmentation("Coarse Dropout"))
        self.ui.dropout_width.sliderReleased.connect(lambda: self.doAugmentation("Coarse Dropout"))
        self.ui.dropout_height.sliderReleased.connect(lambda: self.doAugmentation("Coarse Dropout"))
        self.ui.dropout_p.valueChanged.connect(lambda: self.doAugmentation("Coarse Dropout"))
        self.ui.padding_height.valueChanged.connect(lambda: self.doAugmentation("Padding"))
        self.ui.padding_width.valueChanged.connect(lambda: self.doAugmentation("Padding"))
        self.ui.padding_position_options.currentIndexChanged.connect(lambda: self.doAugmentation("Padding"))
        self.ui.padding_border_options.currentIndexChanged.connect(lambda: self.doAugmentation("Padding"))
        self.ui.padding_p.valueChanged.connect(lambda: self.doAugmentation("Padding"))
        self.ui.gdistort_step.valueChanged.connect(lambda: self.doAugmentation("Grid Distortion"))
        self.ui.gdistort_limit.valueChanged.connect(lambda: self.doAugmentation("Grid Distortion"))
        self.ui.gdistort_inter_options.currentIndexChanged.connect(lambda: self.doAugmentation("Grid Distortion"))
        self.ui.gdistort_p.valueChanged.connect(lambda: self.doAugmentation("Grid Distortion"))

        self.ui.add_augmentation.clicked.connect(self.pipeline_a)
        self.ui.augmentation_folder.clicked.connect(self.getDirA)
        self.ui.augmentation_start.clicked.connect(self.saveAugmentation)

        # preprocessing
        self.ui.pre_next.clicked.connect(self.show_next_image_p)
        self.ui.pre_back.clicked.connect(self.show_previous_image_p)
        self.ui.preprocess_options.currentIndexChanged.connect(self.preprocessUpdate)

        self.ui.gray_check.stateChanged.connect(lambda: self.doPreprocessing("Grayscale"))
        self.ui.resize_height.sliderReleased.connect(lambda: self.doPreprocessing("Resize"))
        self.ui.resize_width.sliderReleased.connect(lambda: self.doPreprocessing("Resize"))
        self.ui.morp_scale.sliderReleased.connect(lambda: self.doPreprocessing("Morphological"))
        self.ui.morp_ops_options.currentIndexChanged.connect(lambda: self.doPreprocessing("Morphological"))
        self.ui.ccrop_width.sliderReleased.connect(lambda: self.doPreprocessing("Center Crop"))
        self.ui.ccrop_height.sliderReleased.connect(lambda: self.doPreprocessing("Center Crop"))
        self.ui.ccrop_pad_check.stateChanged.connect(lambda: self.doPreprocessing("Center Crop"))
        self.ui.ccrop_padpos_options.currentIndexChanged.connect(lambda: self.doPreprocessing("Center Crop"))
        self.ui.ccrop_border_options.currentIndexChanged.connect(lambda: self.doPreprocessing("Center Crop"))
        self.ui.smallest_size.sliderReleased.connect(lambda: self.doPreprocessing("Smallest Max Size"))
        self.ui.smallest_inter_options.currentIndexChanged.connect(lambda: self.doPreprocessing("Smallest Max Size"))

        # self.ui.preprocessing_list.currentRowChanged.connect()

        self.ui.add_preprocess.clicked.connect(self.pipeline_p)
        self.ui.preprocess_folder.clicked.connect(self.getDirP)
        self.ui.preprocessing_start.clicked.connect(self.savePreprocessing)

        # train
        self.ui.train_split.valueChanged.connect(self.modifySplit)
        self.ui.val_split.valueChanged.connect(self.modifySplit)
        self.ui.test_split.valueChanged.connect(self.modifySplit)

        self.ui.train_sources.clicked.connect(self.getTrainSources)
        self.ui.train_start.clicked.connect(self.Train)

        # evaluate
        self.ui.evaluation_start.clicked.connect(self.Evaluate)

        # inference
        self.ui.inference_start.clicked.connect(self.Inference)

        self.ui.inf_next.clicked.connect(self.next_inference)
        self.ui.inf_back.clicked.connect(self.previous_inference)

        # self.ui.preprocessed_check.clicked.connect(self.displayInference)

        self.ui.inference_source_image.clicked.connect(self.updateInferenceImage)
        self.ui.inference_sources_folder.clicked.connect(self.updateInferenceFolder)

        # export
        self.ui.script_save.clicked.connect(self.updateScriptLocation)
        self.ui.model_save.clicked.connect(self.updateModelLocation)
        self.ui.model_export.clicked.connect(self.modelExport)
        self.ui.script_export.clicked.connect(self.scriptExport)
        self.ui.preview.clicked.connect(self.scriptPreview)

    def modifySplit(self):
        train_value = self.ui.train_split.value()
        val_value = self.ui.val_split.value()
        test_value = self.ui.test_split.value()
        
        total = train_value + val_value + test_value

        if total != 100:
            sender = self.sender()

            if sender == self.ui.train_split:
                if train_value + val_value + test_value > 100:
                    if test_value > 0:
                        test_value = max(test_value - (train_value + val_value + test_value - 100), 0)
                        self.ui.test_split.setValue(test_value)
                    if test_value == 0 and train_value + val_value + test_value > 100:
                        val_value = 100 - train_value - test_value
                        self.ui.val_split.setValue(val_value)

            elif sender == self.ui.val_split:
                if train_value + val_value + test_value > 100:
                    if test_value > 0:
                        test_value = max(test_value - (train_value + val_value + test_value - 100), 0)
                        self.ui.test_split.setValue(test_value)
                    if test_value == 0 and train_value + val_value + test_value > 100:
                        train_value = 100 - val_value - test_value
                        self.ui.train_split.setValue(train_value)

            elif sender == self.ui.test_split:
                if train_value + val_value + test_value > 100:
                    if val_value > 0:
                        val_value = max(val_value - (train_value + val_value + test_value - 100), 0)
                        self.ui.val_split.setValue(val_value)
                    if val_value == 0 and train_value + val_value + test_value > 100:
                        train_value = 100 - val_value - test_value
                        self.ui.train_split.setValue(train_value)

    def selectProject(self, index):
        self.click = index

        elements = [self.ui.classification, self.ui.instance, self.ui.detection, self.ui.semantic]
        default_style = "border: 2px solid transparent; border-radius: 5px; background-color: rgb(240, 240, 240);"
        selected_style = "border: 2px solid rgb(227, 0, 52); border-radius: 5px;"
        
        # Loop through all elements and apply the default style
        for i, element in enumerate(elements):
            if i == index:
                element.setStyleSheet(selected_style)
            else:
                element.setStyleSheet(default_style)

    def browse(self):
        self.projectBrowse = QFileDialog.getExistingDirectory(
            self, 
            caption="Choose the project directory", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Downloads"
        )
        if self.projectBrowse is None:
            QMessageBox.warning(self, "Warning", "The project name is empty, please add it!")
        
        if self.projectBrowse:
            self.ui.project_location.setText(self.projectBrowse)

    def createProject(self):
        self.projectName = self.ui.project_name.displayText()
        
        if self.projectName == "":
            QMessageBox.warning(self, "Warning", "The project name is empty, please add it!")
            return

        if self.projectBrowse is None:
            self.browse()

        self.projectFolder = f"{self.projectBrowse}/{self.projectName}"

        if not os.path.exists(self.projectFolder): 
            os.mkdir(self.projectFolder)

        for i in range(2, self.ui.tabWidget.count()):
            self.ui.tabWidget.tabBar().setTabTextColor(i, QColor('grey'))

        self.ui.tabWidget.setTabEnabled(1,True)

        self.ui.tabWidget.setCurrentIndex(self.ui.tabWidget.currentIndex()+1)

        self.info_setup(mode=self.click)

        os.mkdir(f"{self.projectFolder}/outputs")
        self.script_location = f"{self.projectFolder}/outputs/scripts.py"
        self.model_location = f"{self.projectFolder}/outputs/model.onnx"
        self.ui.script_location.setText(self.script_location)
        self.ui.model_location.setText(self.model_location)

        # if self.pipelineMode == 0 or self.pipelineMode == 1:
        #     self.ui.preprocessed_check.setCheckable(False)

    def getTrainSources(self):
        self.trainSources = QFileDialog.getExistingDirectory(
            self, 
            caption="Choose the augmentation directory", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Pictures/"
        )

    def update_detection_training(self):
        image = cv2.imread(f"{self.projectFolder}/yolo/train/results.png")
        qlabel = ImageWidget(image=image)
        self.ui.train_graph_layout_1.addWidget(qlabel)
        image = cv2.imread(f"{self.projectFolder}/yolo/train/PR_curve.png")
        qlabel = ImageWidget(image=image)
        self.ui.train_graph_layout_2.addWidget(qlabel)

    def update_detection_eval(self):
        image = cv2.imread(f"{self.projectFolder}/yolo/validation/confusion_matrix.png")
        qlabel = ImageWidget(image=image)
        self.ui.confusion_matrix_layout.addWidget(qlabel)
        image = cv2.imread(f"{self.projectFolder}/yolo/validation/PR_curve.png")
        qlabel = ImageWidget(image=image)
        self.ui.global_measure_layout.addWidget(qlabel)

    def Train(self):
        
        if self.trainSources is None:
            self.getTrainSources()

        self.ui.train_mainwindow_tab.setCurrentIndex(1)

        match self.click:
            case 0:
                # Loss plot
                self.loss_plot = pg.PlotWidget(title="Loss Plot")
                self.loss_plot.setBackground('w')
                self.loss_curve = self.loss_plot.plot(pen='k')
                self.loss_data = []
                self.loss_epochs = []
                self.ui.train_graph_layout_1.addWidget(self.loss_plot)
                
                # Error rate plot
                self.error_plot = pg.PlotWidget(title="Top-1 Error Rate Plot")
                self.error_plot.setBackground('w')
                self.error_curve = self.error_plot.plot(pen='k')
                self.error_data = []
                self.error_epochs = []
                self.ui.train_graph_layout_2.addWidget(self.error_plot)

                self.classificationModel_n = classificationModel(
                    dataset_dir=self.trainSources,
                    split_ratio=[
                        float(self.ui.train_split.value()/100), 
                        float(self.ui.val_split.value()/100), 
                        float(1-float(self.ui.train_split.value()/100)-float(self.ui.val_split.value()/100))
                    ],
                    backbone=self.ui.nn_pretrained_cls.currentText(),
                    device=getDevice(selected_device=self.ui.nn_device.currentText()),
                    epochs=self.ui.nn_epoch.value(),
                    batch_size=self.ui.nn_batch_size.value(),
                    lr=self.ui.nn_lr.value(),
                    loss_fn=self.ui.loss_fn_options.currentText(),
                    optimizer=self.ui.optimizer_options.currentText()
                )
                
                self.classificationTrainThread = classificationTrainThread(
                    model=self.classificationModel_n.model,
                    device=self.classificationModel_n.device, 
                    epochs=self.classificationModel_n.epochs,
                    train_loader=self.classificationModel_n.train_loader,
                    criterion=self.classificationModel_n.loss_fn,
                    optimizer=self.classificationModel_n.optimizer
                )

                # Connect signals to update methods
                self.classificationTrainThread.loss_signal.connect(self.update_loss)
                self.classificationTrainThread.error_signal.connect(self.update_error)
                self.classificationTrainThread.time_elapsed_signal.connect(self.update_time_elapsed)
                self.classificationTrainThread.time_left_signal.connect(self.update_time_left)
                self.classificationTrainThread.progress_bar.connect(self.updateProgressbar)

                self.updateSourcesNumber()

                self.classificationTrainThread.start()
    
            case 1:
                pass # instance segmentation goes here
        
            case 2:
                self.detectionTrainThread = yoloTrainThread(
                    dataset_dir=self.imagesFolder,
                    split_ratio=None,  
                    device="cpu", 
                    epochs=self.ui.nn_epoch.value(), 
                    batch_size=32, 
                    lr=None, 
                    loss_fn="default", 
                    optimizer="default",
                    imgsz=self.ui.nn_image_width.value(),
                    save_dir=self.projectFolder
                )
                self.detectionTrainThread.start()
                self.ui.train_progress.setValue(100)

                self.detectionTrainThread.finish_signal.connect(self.update_training)

            case 3: 
                pass # semantic segmentation goes here

            case _:
                return

    def updateSourcesNumber(self):
        lengths = self.classificationModel_n.datsetLength
        for i, source in enumerate(['train', 'evaluate', 'test']):
            for suffix in ['_source_eval']:
                label = getattr(self.ui, f"{source}{suffix}")
                label.setText(f"{lengths[i]} images")
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
        for i, source in enumerate(['train', 'test', 'evaluate']):
            for suffix in ['_source_infer']:
                label = getattr(self.ui, f"{source}{suffix}")
                label.setText(f"{lengths[i]} images")
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
    
    def updateSourcesNumber_(self):
        lengths = self.classificationModel_n.datsetLength
        for i, source in enumerate(['train', 'evaluate', 'test']):
            for suffix in ['_source_infer', '_source_eval']:
                label = getattr(self.ui, f"{source}{suffix}")
                label.setText(f"{lengths[i]} images")
                label.setAlignment(Qt.AlignmentFlag.AlignRight)

    def updateProgressbar(self, precent):
        self.ui.train_progress.setValue(precent)

    def update_loss(self, loss):
        epoch = len(self.loss_data) + 1
        self.ui.train_result_epoch.setText(f"{epoch}")
        self.ui.train_result_lr.setText(f"{self.ui.nn_lr.value()}")
        self.ui.train_result_loss.setText(f"{loss:.2f}")
        self.loss_data.append(loss)
        self.loss_epochs.append(epoch)
        self.loss_curve.setData(self.loss_epochs, self.loss_data)

    def update_error(self, error):
        epoch = len(self.error_data) + 1
        self.error_data.append(error)
        self.error_epochs.append(epoch)
        self.error_curve.setData(self.error_epochs, self.error_data)

    def update_time_elapsed(self, time_elapsed):
        self.ui.train_result_te.setText(f"{time_elapsed:.2f}s")

    def update_time_left(self, time_left):
        self.ui.train_result_tl.setText(f"{time_left:.2f}s")

    def classificationTrainDone(self):
        pass

    def getEvalSources(self):
        pass

    def Evaluate(self):
        match self.click:
            case 0:
                self.classificationEvalThread = classificationEvalThread(
                    model=self.classificationModel_n.model,
                    device=self.classificationModel_n.device,
                    valid_data=self.classificationModel_n.valid_dataset,
                    batch_size_eval=self.ui.eval_batch_size.value()
                )

                self.classificationEvalThread.f1.connect(self.updatef1)
                self.classificationEvalThread.precision.connect(self.updatePre)
                self.classificationEvalThread.recall.connect(self.updateRecall)
                self.classificationEvalThread.total_images.connect(self.update_eval_total)
                self.classificationEvalThread.confusion_matrix.connect(self.updateGraph)

                self.classificationEvalThread.start()
                self.ui.evaluate_mainwindow_tab.setCurrentIndex(1)
            
            case 1:
                pass

            case 2:
                self.detectionEvalThread = yoloEvalThread(
                    model=self.detectionTrainThread.model,
                    save_dir=self.projectFolder
                )
                self.detectionEvalThread.start()

                self.detectionEvalThread.finish_signal.connect(self.update_eval)

            case 3:
                pass

            case _:
                return

    def updatef1(self, value):
        self.ui.f1score.setText(f"{value:.2f}")

    def updatePre(self, value):
        self.ui.mean_precison.setText(f"{value:.2f}")
    
    def updateRecall(self, value):
        self.ui.recall.setText(f"{value:.2f}")
    
    def update_eval_total(self, value):
        self.ui.evaluate_image.setText(f"{value:.2f}")
        
    def updateGraph(self, array):
        self.canvas = ConfusionMatrixCanvas()
        self.ui.confusion_matrix_layout.addWidget(self.canvas)
        self.canvas.plot_confusion_matrix(array, self.classificationModel_n.classes)

        true_total_predictions = np.sum(array)
        true_predictions = np.trace(array)
        false_predictions = true_total_predictions - true_predictions
        self.pieChart(true_predictions, false_predictions)

    def pieChart(self, true_predictions, false_predictions):
        self.series = QPieSeries()

        self.series.append('True Predictions', true_predictions)
        self.series.append('False Predictions',false_predictions)

        for slice in self.series.slices():
            slice.setLabelVisible()
            slice.setLabel(f'{slice.label()} ({slice.value()})')

            # Set properties for the slice named 'Joe'
            if slice.label() == 'True Predictions':
                slice.setExploded()
                slice.setPen(QPen(Qt.GlobalColor.darkGreen, 2))
                slice.setBrush(Qt.GlobalColor.green)

        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.setTitle('Total Images Predictions')
        self.chart.legend().hide()

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.ui.global_measure_layout.addWidget(self.chart_view)


    def Inference(self):

        match self.click:
        
            case 0:
                self.ui.inf_next.setEnabled(True)
                self.ui.inf_back.setEnabled(True)

                self.all_image_paths = []
                self.all_labels = []
                self.all_preds = []
                self.all_confidences = []

                self.all_image_paths.clear()
                self.all_labels.clear()
                self.all_preds.clear()
                self.all_confidences.clear()

                if self.inferenceMode == "dataloader":
                    inference_data = DataLoader(self.classificationModel_n.test_dataset, batch_size=1)
                elif self.inferenceMode == "folder":
                    inference_data = self.inference_folder
                    self.inferenceMode = "folder_inferenced"
                elif self.inferenceMode == "image":
                    inference_data = self.inference_image
                    self.ui.inf_next.setEnabled(False)
                    self.ui.inf_back.setEnabled(False)

                self.classificationInferenceThread = classificationInferenceThread_test(
                    model=self.classificationModel_n.model,
                    device=self.classificationModel_n.device,
                    input_data=inference_data,
                )

                self.classificationInferenceThread.inference_finished.connect(self.updateInference)

                self.classificationInferenceThread.start()
            
            case 1:
                pass # instance segmentation goes here
    
            case 2:
                self.yoloInferenceThread = yoloInferThread(
                    model=self.detectionTrainThread.model,
                    save_dir=self.projectFolder,
                    source=self.inference_image
                )

            case 3: 
                pass # semantic segmentation goes here

            case _:
                return


    def updateInferenceImage(self):
        self.inference_image, _ = QFileDialog.getOpenFileName(
            self, 
            caption="Choose an image to inference", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Pictures/"
        )

        if self.inference_image == "":
            return

        self.inferenceMode = "image"

        image = cv2.imread(self.inference_image)
        self.inf_image_widget = ImageWidget(image=image)
        self.ui.image_inf_widget.setWidget(self.inf_image_widget)
        self.ui.inf_image_name.setText(self.inference_image)

    def updateInferenceFolder(self):
        self.inference_folder = QFileDialog.getExistingDirectory(
            self, 
            caption="Choose an image folder to inference", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Pictures/"
        )

        if self.inference_folder == "":
            return

        self.inferenceMode = "folder"

        self.all_image_paths_folder_inference = []

        for image_file in os.listdir(self.inference_folder):
            image_path = os.path.join(self.inference_folder, image_file)
            self.all_image_paths_folder_inference.append(image_path)

        self.updateInferenceFolderSwitch()

    def updateInferenceFolderSwitch(self):
        image = cv2.imread(self.all_image_paths_folder_inference[self.current_inf_index_folder])
        self.inf_image_widget = ImageWidget(image=image)
        self.ui.image_inf_widget.setWidget(self.inf_image_widget)
        self.ui.inf_image_id.setText(f'{self.current_inf_index_folder + 1} / {len(self.all_image_paths_folder_inference)}')
        self.ui.inf_image_name.setText(self.all_image_paths_folder_inference[self.current_inf_index_folder])

    def updateInference(self, results):
        if self.inferenceMode == "image":
            self.ui.inference_label.setText("None")
            self.ui.inference_prediction.setText(f"{self.classificationModel_n.classes[results['prediction']]}")
            self.ui.inference_score.setText(f"{results['confidence']:.2f}")
        else:
            for result in results:
                self.all_image_paths.append(result['image_path'])
                self.all_labels.append(result['label'])
                self.all_preds.append(result['prediction'])
                self.all_confidences.append(result['confidence'])

            self.displayInference()

    def displayInference(self):
        image = cv2.imread(self.all_image_paths[self.current_inf_index])

        # if self.ui.preprocessed_check.isChecked():
        #     pass

        self.inf_image_widget = ImageWidget(image=image)
        self.ui.image_inf_widget.setWidget(self.inf_image_widget)
        self.ui.inf_image_id.setText(f'{self.current_inf_index + 1} / {len(self.all_image_paths)}')
        self.ui.inf_image_name.setText(self.all_image_paths[self.current_inf_index])

        if self.inferenceMode == "dataloader":
            self.ui.inference_label_classification.setText(f"{self.classificationModel_n.classes[self.all_labels[self.current_inf_index]]}")
            self.ui.inference_prediction_classification.setText(f"{self.classificationModel_n.classes[self.all_preds[self.current_inf_index]]}")
            self.ui.inference_score_classification.setText(f"{self.all_confidences[self.current_inf_index]:.2f}")
        else:
            self.ui.inference_label.setText("None")
            self.ui.inference_prediction.setText(f"{self.classificationModel_n.classes[self.all_preds[self.current_inf_index_folder]]}")
            self.ui.inference_score.setText(f"{self.all_confidences[self.current_inf_index_folder]:.2f}")

    def displayPreprocessed(self):
        image = cv2.imread(self.all_image_paths[self.current_inf_index])
        self.ui.image_inf_widget.setWidget(self.inf_image_widget)

    def next_inference(self):
        if self.inferenceMode == "dataloader":
            self.current_inf_index = (self.current_inf_index + 1) % len(self.all_image_paths)
            self.displayInference()
        elif self.inferenceMode == "folder":
            self.current_inf_index_folder = (self.current_inf_index_folder + 1) % len(self.all_image_paths_folder_inference)
            self.updateInferenceFolderSwitch()
        elif self.inferenceMode == "folder_inferenced":
            self.current_inf_index_folder = (self.current_inf_index_folder + 1) % len(self.all_image_paths_folder_inference)
            self.updateInferenced()

    def previous_inference(self):
        if self.inferenceMode == "dataloader":
            self.current_inf_index = (self.current_inf_index - 1) % len(self.all_image_paths)
            self.displayInference()
        elif self.inferenceMode == "folder":
            self.current_inf_index_folder = (self.current_inf_index_folder - 1) % len(self.all_image_paths_folder_inference)
            self.updateInferenceFolderSwitch()
        elif self.inferenceMode == "folder_inferenced":
            self.current_inf_index_folder = (self.current_inf_index_folder - 1) % len(self.all_image_paths_folder_inference)
            self.updateInferenced()

    def updateInferenced(self):
        image = cv2.imread(self.all_image_paths[self.current_inf_index_folder])
        
        self.inf_image_widget = ImageWidget(image=image)
        self.ui.image_inf_widget.setWidget(self.inf_image_widget)
        self.ui.inf_image_id.setText(f'{self.current_inf_index_folder + 1} / {len(self.all_image_paths)}')
        self.ui.inf_image_name.setText(self.all_image_paths[self.current_inf_index_folder])
        
        self.ui.inference_label.setText("None")
        self.ui.inference_prediction.setText(f"{self.classificationModel_n.classes[self.all_preds[self.current_inf_index_folder]]}")
        self.ui.inference_score.setText(f"{self.all_confidences[self.current_inf_index_folder]:.2f}")

    def updateScriptLocation(self):
        self.script_location, _ = QFileDialog.getSaveFileName(
            self, 
            caption="Script Filename", 
            directory=self.projectFolder, 
            filter="Python Files (*.py)"
        )
        if self.script_location == "":
            self.script_location = f"{self.projectFolder}/outputs/scripts.py"

        self.ui.script_location.setText(self.script_location)

    def scriptExport(self):

        with open("./templates/pytorch/classification.py", "r") as template_file:
            template_content = template_file.read()

        self.script = generate_script(
            template_content=template_content,
            dataset_dir="",
            split_ratio=[0.7,0.2,0.1],
            pretrained="",
            epochs=self.ui.nn_epoch.value(),
            batch_size=self.ui.nn_batch_size.value(),
            learning_rate=self.ui.nn_lr.value(),
            device="cpu",
            augmentation_pipeline=self.augmentations,
            preprocessing_pipeline=self.preprocesses
        )

        with open(self.script_location, "w") as output_file:
            output_file.write(self.script)

        QMessageBox.information(self, "Notification", f"Your script is saved at {self.script_location}")
        
    def scriptPreview(self, content):
        with open("./templates/pytorch/classification.py", "r") as template_file:
            template_content = template_file.read()

        self.script = generate_script(
            template_content=template_content,
            dataset_dir="",
            split_ratio=[0.7,0.2,0.1],
            pretrained="",
            epochs=self.ui.nn_epoch.value(),
            batch_size=self.ui.nn_batch_size.value(),
            learning_rate=self.ui.nn_lr.value(),
            device="cpu",
            augmentation_pipeline=self.augmentations,
            preprocessing_pipeline=self.preprocesses
        )

        self.editor = QCodeEditor()
        self.editor.setHighlighter(QPythonHighlighter())
        self.editor.resize(800, 600)
        self.editor.setPlainText(self.script)
        self.editor.setReadOnly(True)
        self.editor.setWindowTitle("Code Preview")
        self.editor.setWindowIcon(QIcon("C:/Users/huyhoang.nguyen@infineon.com/Desktop/DLGUI/ui/resources/generative.png"))
        self.editor.show()

    def updateModelLocation(self):
        self.model_location, _ = QFileDialog.getSaveFileName(
            self, 
            caption="Model Filename", 
            directory=self.projectFolder, 
            filter="ONNX Files (*.onnx);;H5 Files (*.h5)"
        )

        if self.model_location == "":
            self.model_location = f"{self.projectFolder}/outputs/model.onnx"

        self.ui.model_location.setText(self.model_location)

    def modelExport(self):
        self.classificationModel_n.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)  # Example input size (batch_size=1, channels=3, height=224, width=224)

        # Step 4: Export the Model to ONNX
        onnx_file_path = self.model_location
        torch.onnx.export(
            self.classificationModel_n.model,                     # model being run
            dummy_input,               # model input (or a tuple for multiple inputs)
            onnx_file_path,            # where to save the model (can be a file or file-like object)
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=12,          # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['input'],     # the model's input names
            output_names=['output'],   # the model's output names
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # variable length axes
        )

        QMessageBox.information(self, "Notification", f"Your model is saved at {self.model_location}") 
        
    def doPreprocessing(self, method):
        if method == "Grayscale":
            if self.ui.gray_check.isChecked():
                self.transformP = A.ToGray(p=1.0)
            else:
                self.transformP = A.ToGray(p=0.0)
        elif method == "Resize":
            self.transformP = A.Resize(width=self.ui.resize_width.value(), height=self.ui.resize_height.value(), p=1.0)
        elif method == "Morphological":
            self.transformP = A.Morphological(scale=self.ui.morp_scale.value(), operation=self.ui.morp_ops_options.currentText(), p=1.0)
        elif method == "Center Crop":
            if self.ui.ccrop_pad_check.isChecked():
                self.transformP = A.CenterCrop(
                    width=self.ui.ccrop_width.value(), 
                    height=self.ui.ccrop_height.value(), 
                    pad_if_needed = True,
                    pad_position = self.ui.ccrop_padpos_options.currentText(),
                    border_mode = borderFlag(literal=self.ui.ccrop_border_options.currentText()),
                    p=1.0
                )
            else:
                self.transformP = A.CenterCrop(width=self.ui.ccrop_width.value(), height=self.ui.ccrop_height.value(), p=1.0)
        elif method == "Smallest Max Size":
            self.transformP = A.SmallestMaxSize(
                max_size=self.ui.smallest_size.value(), 
                interpolation=interpolationFlag(literal=self.ui.smallest_inter_options.currentText()), 
                p=1.0
            )

        image = cv2.imread(self.paths_p[self.current_index_p])

        if self.preprocesses is not None:
            for step in self.preprocesses:
                transformed = step(image=image)
                image = transformed["image"]
        transformed = self.transformP(image=image)
        self.pre_image_widget = ImageWidget(image=transformed["image"])
        self.ui.image_pre_widget.setWidget(self.pre_image_widget)

    def pipeline_p(self):
        
        method = self.ui.preprocess_options.currentText()

        if self.transformP is None:
            self.doPreprocessing(method)

        self.preprocesses.append(self.transformP)

        if method in self.preprocessesList:
            QMessageBox.warning(self, "Warning", "You've already added this preprocessing method!")
        else:
            listItem = QListWidgetItem(method)
            removeButton = QToolButton()
            removeIcon = QIcon(QPixmap('C:/Users/huyhoang.nguyen@infineon.com/Desktop/DLGUI/ui/resources/close.png').scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            removeButton.setIcon(removeIcon)
            removeButton.setAutoRaise(True)
            removeButton.setIconSize(QSize(20, 20))
            removeButton.setFixedSize(20, 20)
            removeButton.clicked.connect(lambda: self.removeMethodP(listItem, method))
            listItem.setSizeHint(removeButton.sizeHint())

            self.ui.preprocessing_list.addItem(listItem)
            self.preprocessesList.add(method)

    def removeMethodP(self, listItem, method):
        row = self.ui.preprocessing_list.row(listItem)
        self.ui.preprocessing_list.takeItem(row)
        self.preprocessesList.remove(method)

    def stack_control(self):
        self.update_image()

        self.ui.augmentation_stack.setCurrentIndex(self.ui.augmentation_options.currentIndex())
    
    def preprocessUpdate(self):
        if self.current_index_p:
            self.update_image_p()
        self.ui.preprocess_stack.setCurrentIndex(self.ui.preprocess_options.currentIndex())
        self.doPreprocessing(method=self.ui.preprocess_options.currentText())

    def augmentation_changed(self):
        pass

    def doAugmentation(self, method):
        if method == "Crop":
            self.transformA = A.Crop(
                x_min=0, 
                y_min=0, 
                x_max=self.ui.crop_width_a.value(), 
                y_max=self.ui.crop_height_a.value(), 
                p=self.ui.crop_p.value()
            )
        elif method == "Gauss Noise":
            self.transformA = A.GaussNoise(
                std_range=(0.2, 0.44),
                mean_range=(-0.2, 0.2),
                per_channel=self.ui.noise_channel_check.isChecked(),
                noise_scale_factor=0.5,
                p=self.ui.noise_p.value()
            )
        elif method == "Rotate":
            self.transformA = A.Rotate(
                limit=self.ui.rotate_limit.value(),
                interpolation=interpolationFlag(self.ui.rotate_inter_options.currentText()),
                border_mode=borderFlag(self.ui.rotate_border_options.currentText()),
                p=self.ui.rotate_p.value()
            )
        elif method == "Horizontal Flip":
            if self.ui.horizontal_check.isChecked():
                self.transformA = A.HorizontalFlip(p=self.ui.horizontal_p.value())
            else:
                self.transformA = A.HorizontalFlip(p=0)
        elif method == "Vertical Flip":
            if self.ui.vertical_check.isChecked():
                self.transformA = A.VerticalFlip(p=self.ui.vertical_p.value())
            else:
                self.transformA = A.VerticalFlip(p=0)
        elif method == "Coarse Dropout":
            self.transformA = A.CoarseDropout(
                num_holes_range=(1, self.ui.dropout_holes.value()),
                hole_width_range=(3, self.ui.dropout_width.value()),
                hole_height_range=(3, self.ui.dropout_height.value()),
                p=self.ui.dropout_p.value()
            )
        elif method == "Padding":
            self.transformA = A.PadIfNeeded(
                min_height=self.ui.padding_height.value(),
                min_width=self.ui.padding_width.value(),
                position=self.ui.padding_position_options.currentText(),
                border_mode=borderFlag(self.ui.padding_border_options.currentText()),
                p=self.ui.padding_p.value()
            )
        elif method == "Grid Distortion":
            self.transformA = A.GridDistortion(
                num_steps=self.ui.gdistort_step.value(),
                distort_limit=self.ui.gdistort_limit.value(),
                interpolation=interpolationFlag(self.ui.gdistort_inter_options.currentText()),
                p=self.ui.gdistort_p.value()
            )

        image = cv2.imread(self.paths[self.current_index])

        transformed = self.transformA(image=image)
        self.aug_image_widget = ImageWidget(image=transformed["image"])
        self.ui.image_aug_widget.setWidget(self.aug_image_widget)

    def pipeline_a(self):

        method = self.ui.augmentation_options.currentText()

        if self.transformA is None:
            self.doAugmentation(method)
        
        self.augmentations.transforms.append(self.transformA)
        
        if method in self.augmentationsList:
            QMessageBox.warning(self, "Warning", "You've already added this augmentation method!")
        else:
            listItem = QListWidgetItem(method)
            removeButton = QToolButton()
            removeIcon = QIcon(QPixmap('C:/Users/huyhoang.nguyen@infineon.com/Desktop/DLGUI/ui/resources/close.png').scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            removeButton.setIcon(removeIcon)
            removeButton.setAutoRaise(True)
            removeButton.setIconSize(QSize(20, 20))
            removeButton.setFixedSize(20, 20)
            removeButton.clicked.connect(lambda: self.removeMethod(listItem, method))
            listItem.setSizeHint(removeButton.sizeHint())

            self.ui.augmentation_list.addItem(listItem)
            self.augmentationsList.add(method)

    def removeMethod(self, listItem, method):
        row = self.ui.augmentation_list.row(listItem)
        self.ui.augmentation_list.takeItem(row)
        self.augmentationsList.remove(method)

    def getDirA(self):
        self.augmentedFolder = QFileDialog.getExistingDirectory(
            self, 
            caption="Choose the augmentation directory", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Pictures/"
        )
        
        if self.augmentedFolder == "":
            self.augmentedFolder = self.projectFolder
        
        self.ui.statusbar.showMessage(f"Augmentation folder will be saved at {self.augmentedFolder}")

    def getDirP(self):
        self.preprocessedFolder = QFileDialog.getExistingDirectory(
            self, 
            caption="Choose the preprocessed directory", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Pictures/"
        )

        if self.preprocessedFolder == "":
            self.preprocessedFolder = self.projectFolder

        self.ui.statusbar.showMessage(f"Augmentation folder will be saved at {self.preprocessedFolder}")

    def saveAugmentation(self):
        augmentationFolder(
            dataset_path=self.imagesFolder,
            save_path=self.augmentedFolder,
            transform_method=self.augmentations,
            number=self.ui.augmentation_number.value()
        )

        dataSource = f"{self.augmentedFolder}/AugmentedDataset"

        # Preprocessing tab
        if self.pipelineMode == 1:
            dataSource = self.imagesFolder

        self.image_names_p, self.labels_p, self.paths_p = load_images_and_labels(dataSource)
        image = cv2.imread(self.paths_p[self.current_index_p])

        # Label display
        id = self.all_classes.index(self.labels_p[self.current_index_p])
        self.ui.preprocess_classes_list.setCurrentRow(id)

        self.pre_image_widget = ImageWidget(image=image)
        self.ui.image_pre_widget.setWidget(self.pre_image_widget)
        self.ui.pre_image_id.setText(f'{self.current_index_p + 1} / {len(self.paths_p)}')
        self.ui.pre_image_name.setText(self.image_names_p[self.current_index_p])

        QMessageBox.information(self, "Notification", "Augmentation Completed!")

        self.trainSources = f"{self.projectFolder}/AugmentedDataset"

        if self.pipelineMode == 2:
            for i in range(self.ui.tabWidget.currentIndex(), self.ui.tabWidget.count()):
                self.ui.tabWidget.setTabEnabled(i, True)
        else: 
            self.ui.tabWidget.setTabEnabled(self.ui.tabWidget.currentIndex()+1, True)
            self.ui.tabWidget.setCurrentIndex(self.ui.tabWidget.currentIndex()+1)

    def savePreprocessing(self):
        if self.pipelineMode == 0:
            dataSource = f"{self.augmentedFolder}/AugmentedDataset"
        elif self.pipelineMode == 1:
            dataSource = self.imagesFolder

        preprocessedFolder(
            dataset_path=dataSource,
            save_path=self.preprocessedFolder,
            transform_method=self.preprocesses,
        )

        self.ui.tabWidget.setCurrentIndex(self.ui.tabWidget.currentIndex()+1)
        QMessageBox.information(self, "Notification", "Preprocessing Completed!")

        self.trainSources = f"{self.projectFolder}/PreprocessedDataset"

        for i in range(self.ui.tabWidget.currentIndex(), self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, True)

    def preprocess_changed(self):
        pass

    def DatasetInit(self):
        self.imagesFolder = QFileDialog.getExistingDirectory(
            self, 
            caption="Choose the images folder", 
            directory="C:/Users/huyhoang.nguyen@infineon.com/Pictures/"
        )

        if self.imagesFolder is None:
            return

        match self.click:
            case 0:
                self.ui.dataset_widget.setLayout(gridInit(self.imagesFolder))

                all_classes = []
                self.all_classes = all_classes
                for subdir in os.listdir(self.imagesFolder):
                    all_classes.append(subdir)

                self.add_classes_to_list_widget(self.ui.dataset_classes_list, all_classes)
                self.add_classes_to_list_widget(self.ui.augmentation_classes_list, all_classes)
                self.add_classes_to_list_widget(self.ui.preprocess_classes_list, all_classes)

                if self.pipelineMode == 3:
                    for i in range(self.ui.tabWidget.currentIndex()+1, self.ui.tabWidget.count()):
                        self.ui.tabWidget.setTabEnabled(i, True)
                else:
                    self.ui.tabWidget.setTabEnabled(self.ui.tabWidget.currentIndex()+1, True)

                # Set (augmentation/preprocessing) image
                self.display()

            case 1:
                pass # instance segmentation goes here
                print("case1")
    
            case 2:
                self.ui.dataset_widget.setLayout(yoloInit(self.imagesFolder))

                if self.pipelineMode == 3:
                    for i in range(self.ui.tabWidget.currentIndex()+1, self.ui.tabWidget.count()):
                        self.ui.tabWidget.setTabEnabled(i, True)
                else:
                    self.ui.tabWidget.setTabEnabled(self.ui.tabWidget.currentIndex()+1, True)

            case 3: 
                pass # semantic segmentation goes here
                print("case3")

            case _:
                print("caseNone")
                return
    

    def add_classes_to_list_widget(self, widget, class_list):
        for i, classname in enumerate(class_list):
            item = QListWidgetItem()
            item.setText(classname)
            item.setIcon(self.create_colored_square_icon(colors[i]))
            widget.addItem(item)

    def classClicked(self, class_obj):
        pass

    def dropTab(self):
        selected_button = self.ui.buttonGroup.checkedButton()
        selection = self.ui.buttonGroup.id(selected_button) + 5

        if selection == 0:
            self.pipelineMode = 0
            self.augmentedFolder = self.projectFolder
            self.preprocessedFolder = self.projectFolder
            self.configListWidget()
        # preprocessing only
        elif selection == 1: 
            self.ui.tabWidget.removeTab(self.ui.tabWidget.currentIndex()+2) 
            self.pipelineMode = 1
            self.preprocessedFolder = self.projectFolder
            self.configListWidget()
        # augmentation only
        elif selection == 2:
            self.ui.tabWidget.removeTab(self.ui.tabWidget.currentIndex()+3)
            self.pipelineMode = 2
            self.augmentedFolder = self.projectFolder
        elif selection == 3:
            self.ui.tabWidget.removeTab(self.ui.tabWidget.currentIndex()+2)
            self.ui.tabWidget.removeTab(self.ui.tabWidget.currentIndex()+2)
            self.pipelineMode = 3
            self.trainSources = self.imagesFolder

        self.ui.tabWidget.setTabEnabled(2, True)

        self.ui.tabWidget.setCurrentIndex(self.ui.tabWidget.currentIndex()+1)

    def configListWidget(self):
        self.ui.preprocessing_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ui.preprocessing_list.setDragEnabled(True)
        self.ui.preprocessing_list.setAcceptDrops(True)
        self.ui.preprocessing_list.setDropIndicatorShown(True)
        self.ui.preprocessing_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

    def det_dataset(self):
        pass

    def seg_dataset(self):
        pass

    def info_setup(self, mode):
        # set info stacks
        self.ui.dataset_panel_info.setCurrentIndex(mode)
        self.ui.pipeline_panel_info.setCurrentIndex(mode)
        self.ui.augment_panel_info.setCurrentIndex(mode)
        self.ui.preprocess_panel_info.setCurrentIndex(mode)
        self.ui.train_panel_info.setCurrentIndex(mode)
        self.ui.evaluate_panel_info.setCurrentIndex(mode)
        self.ui.inference_panel_info.setCurrentIndex(mode)
        self.ui.export_panel_info.setCurrentIndex(mode)

        # set other stacks
        self.ui.train_setting_nn_model_stack.setCurrentIndex(mode)
        self.ui.train_setting_params_stack.setCurrentIndex(mode)
        self.ui.inference_result_stack.setCurrentIndex(mode)
        self.ui.augmentation_stack.setCurrentIndex(0)
        self.ui.preprocess_stack.setCurrentIndex(0)

        # set tabs
        self.ui.evaluate_mainwindow_tab.setCurrentIndex(0)
        self.ui.train_mainwindow_tab.setCurrentIndex(0)

        # set combo box
        self.ui.augmentation_options.setCurrentIndex(0)
        self.ui.preprocess_options.setCurrentIndex(0)

        # set class panel
        
    def display(self):
        self.image_names, self.labels, self.paths = load_images_and_labels(self.imagesFolder)
        image = cv2.imread(self.paths[self.current_index])

        # Label display
        id = self.all_classes.index(self.labels[self.current_index])
        self.ui.augmentation_classes_list.setCurrentRow(id)

        # Augmentation tab
        self.aug_image_widget = ImageWidget(image=image)
        self.ui.image_aug_widget.setWidget(self.aug_image_widget)
        self.ui.aug_image_id.setText(f'{self.current_index + 1} / {len(self.paths)}')
        self.ui.aug_image_name.setText(self.image_names[self.current_index])

    def show_next_image(self):
        if self.paths:
            self.current_index = (self.current_index + 1) % len(self.paths)
            id = self.all_classes.index(self.labels[self.current_index])
            self.ui.augmentation_classes_list.setCurrentRow(id)
            self.update_image()

    def show_previous_image(self):
        if self.paths:
            self.current_index = (self.current_index - 1) % len(self.paths)
            id = self.all_classes.index(self.labels[self.current_index])
            self.ui.augmentation_classes_list.setCurrentRow(id)
            self.update_image()

    def show_next_image_p(self):
        if self.paths_p:
            self.current_index_p = (self.current_index_p + 1) % len(self.paths_p)
            id = self.all_classes.index(self.labels_p[self.current_index_p])
            self.ui.preprocess_classes_list.setCurrentRow(id)
            self.update_image_p()

    def show_previous_image_p(self):
        if self.paths_p:
            self.current_index_p = (self.current_index_p - 1) % len(self.paths_p)
            id = self.all_classes.index(self.labels_p[self.current_index_p])
            self.ui.preprocess_classes_list.setCurrentRow(id)
            self.update_image_p()

    def update_image(self):
        if self.paths:
            image = cv2.imread(self.paths[self.current_index])
            # Augmentation tab
            self.aug_image_widget = ImageWidget(image=image)
            self.ui.image_aug_widget.setWidget(self.aug_image_widget)
            self.ui.aug_image_id.setText(f'{self.current_index + 1} / {len(self.paths)}')
            self.ui.aug_image_name.setText(self.image_names[self.current_index])

    def update_image_p(self):
        # Preprocessing tab
        if self.paths_p:
            image = cv2.imread(self.paths_p[self.current_index_p])
            if self.preprocesses is not None:
                for step in self.preprocesses:
                    transformed = step(image=image)
                    image = transformed["image"]
            self.pre_image_widget = ImageWidget(image=image)
            self.ui.image_pre_widget.setWidget(self.pre_image_widget)
            self.ui.pre_image_id.setText(f'{self.current_index_p + 1} / {len(self.paths_p)}')
            self.ui.pre_image_name.setText(self.image_names_p[self.current_index_p])

            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow()
    sys.exit(app.exec())