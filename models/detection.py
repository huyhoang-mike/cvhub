from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from typing import Tuple, Any
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torchvision.models import (
    resnet50, alexnet, googlenet, vgg19, inception_v3, efficientnet_b0,
    ResNet50_Weights, AlexNet_Weights, GoogLeNet_Weights, VGG19_Weights, Inception_V3_Weights, EfficientNet_B0_Weights
)
import torch.nn.functional as F
from PIL import Image
import os
from ultralytics import YOLO


class yoloModel:
    def __init__(
            self,
            dataset_dir,
            split_ratio,  
            device, 
            epochs, 
            batch_size, 
            lr, 
            loss_fn, 
            optimizer,
            imgsz,
    ):
        self.dataset_dir = dataset_dir
        self.split_ratio = split_ratio
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.imgsz = imgsz
        
class yoloTrainThread(QThread):

    finish_signal = pyqtSignal(int)

    def __init__(
            self,
            dataset_dir,
            split_ratio,  
            device, 
            epochs, 
            batch_size, 
            lr, 
            loss_fn, 
            optimizer,
            imgsz,
            save_dir
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split_ratio = split_ratio
        self.device = device
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.imgsz = imgsz
        self.save_dir = save_dir

    @pyqtSlot()
    def run(self):
        
        os.chdir(self.save_dir)

        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        model.train(
            data=f"{self.dataset_dir}/data.yaml", 
            epochs=self.epochs, 
            imgsz=self.imgsz,
            project="yolo",
            name="train", 
            plots=True
        )

        self.model = model

        self.finish_signal.emit(1)

        os.chdir(os.getcwd())

class yoloEvalThread(QThread):

    finish_signal = pyqtSignal(int)

    def __init__(self, model, save_dir):
        super().__init__()
        self.model = model
        self.save_dir = save_dir

    @pyqtSlot()
    def run(self):

        os.chdir(self.save_dir)

        self.model.val(
            project="yolo",
            name="validation",
        )
        
        self.finish_signal.emit(1)

        os.chdir(os.getcwd())

class yoloInferThread(QThread):

    def __init__(self, model, save_dir, source):
        super().__init__()
        self.model = model
        self.save_dir = save_dir
        self.source = source

    @pyqtSlot()
    def run(self):

        os.chdir(self.save_dir)

        self.model.predict(self.source, save=True, imgsz=640, conf=0.5)

        os.chdir(os.getcwd())