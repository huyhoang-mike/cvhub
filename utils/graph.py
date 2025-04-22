from PyQt6.QtWidgets import QMainWindow, QApplication, QSizePolicy
import sys
import pyqtgraph as pg
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from torchvision import transforms, datasets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtCharts import QChart, QChartView, QPieSeries

class ConfusionMatrixCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
    
    def plot_confusion_matrix(self, conf_matrix, class_names):
        self.ax.clear()  # Clear the previous plot
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=self.ax)
        self.ax.set_xlabel('Predicted')
        self.ax.set_ylabel('True')
        self.ax.set_title('Confusion Matrix')
        self.fig.tight_layout()  # Adjust layout to ensure everything fits
        self.draw()