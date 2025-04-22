import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import umap
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QGridLayout, QDialog, QProgressBar, QMessageBox, QInputDialog, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
from PyQt6.QtGui import QPixmap
from PIL import Image
from collections import Counter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

class EmbeddingThread(QThread):
    embedding_finished = pyqtSignal(dict)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.progress_message = ""

    def run(self):
        self.process_events("Loading image paths and labels...")
        image_paths, labels = self.load_image_paths_and_labels(self.dataset_path)

        self.process_events("Extracting features...")
        features = self.extract_features(image_paths)

        self.process_events("Calculating similarity search...")
        similarity_results = self.similarity_search(features, image_paths)

        self.embedding_finished.emit({
            "image_paths": image_paths,
            "labels": labels,
            "features": features,
            "similarity_results": similarity_results
        })

    def process_events(self, message):
        self.progress_message = message
        QCoreApplication.processEvents()

    def load_image_paths_and_labels(self, dataset_path):
        image_paths = []
        labels = []
        class_mapping = {}
        class_index = 0
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.split('.')[-1].lower() in {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}:
                    image_paths.append(os.path.join(root, file))
                    class_name = os.path.basename(root)
                    if class_name not in class_mapping:
                        class_mapping[class_name] = class_index
                        class_index += 1
                    labels.append(class_mapping[class_name])
        return image_paths, labels

    def extract_features(self, image_paths):
        model = VGG16(weights='imagenet')
        model = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
        features = []
        for idx, img_path in enumerate(image_paths):
            img_data = self.preprocess_image(img_path)
            feature = model.predict(img_data)
            features.append(feature.flatten())

            # Periodically process events to keep UI responsive
            if idx % 10 == 0:
                self.process_events(f"Extracting features: {idx}/{len(image_paths)}")

        return np.array(features)

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data

    def similarity_search(self, features, image_paths, query_index=0, top_k=5):
        similarity_results = []
        query_feature = features[query_index].reshape(1, -1)
        similarities = cosine_similarity(query_feature, features).flatten()
        top_indices = similarities.argsort()[-top_k-1:-1][::-1]  # Exclude the query itself
        for index in top_indices:
            similarity_results.append(image_paths[index])
        return similarity_results

class EmbeddingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dataset Embedding")
        self.setGeometry(100, 100, 1200, 800)

        self.layout = QVBoxLayout()

        self.dataset_path_label = QLabel("Dataset Path: Not Selected")
        self.layout.addWidget(self.dataset_path_label)

        self.select_button = QPushButton("Select Dataset")
        self.select_button.clicked.connect(self.select_dataset)
        self.layout.addWidget(self.select_button)

        self.analyze_button = QPushButton("Analyze Dataset")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.layout.addWidget(self.analyze_button)

        self.figure_tsne = plt.figure()
        self.canvas_tsne = FigureCanvas(self.figure_tsne)
        self.toolbar_tsne = NavigationToolbar(self.canvas_tsne, self)
        self.tsne_layout = QVBoxLayout()
        self.tsne_layout.addWidget(self.canvas_tsne)
        self.tsne_layout.addWidget(self.toolbar_tsne)
        self.tsne_widget = QWidget()
        self.tsne_widget.setLayout(self.tsne_layout)

        self.figure_umap = plt.figure()
        self.canvas_umap = FigureCanvas(self.figure_umap)
        self.toolbar_umap = NavigationToolbar(self.canvas_umap, self)
        self.umap_layout = QVBoxLayout()
        self.umap_layout.addWidget(self.canvas_umap)
        self.umap_layout.addWidget(self.toolbar_umap)
        self.umap_widget = QWidget()
        self.umap_widget.setLayout(self.umap_layout)

        self.similarity_label = QLabel("Top 5 Similar Images")
        self.similarity_layout = QVBoxLayout()
        self.similarity_layout.addWidget(self.similarity_label)
        self.similarity_widget = QWidget()
        self.similarity_widget.setLayout(self.similarity_layout)

        self.tabs_layout = QGridLayout()
        self.tabs_layout.addWidget(self.tsne_widget, 0, 0)
        self.tabs_layout.addWidget(self.umap_widget, 0, 1)
        self.tabs_layout.addWidget(self.similarity_widget, 1, 0, 1, 2)
        self.tabs_widget = QWidget()
        self.tabs_widget.setLayout(self.tabs_layout)
        self.layout.addWidget(self.tabs_widget)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.dataset_path = None

    def select_dataset(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        self.dataset_path_label.setText(f"Dataset Path: {self.dataset_path}")

    def start_analysis(self):
        if not self.dataset_path:
            self.dataset_path_label.setText("Dataset Path: Not Selected")
            return

        # Disable main window
        self.centralWidget().setEnabled(False)

        # Progress dialog
        self.progress_dialog = QDialog(self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setModal(True)
        layout = QVBoxLayout()
        self.progress_label = QLabel("The application is processing the dataset, it will take some time.")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.progress_dialog.setLayout(layout)
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.show()

        # Start embedding thread
        self.embedding_thread = EmbeddingThread(self.dataset_path)
        self.embedding_thread.embedding_finished.connect(self.on_embedding_finished)
        self.embedding_thread.start()

        # Start a timer to periodically update the progress label
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_label)
        self.progress_timer.start(100)  # Adjust the interval as needed

    def update_progress_label(self):
        self.progress_label.setText(self.embedding_thread.progress_message)
        QCoreApplication.processEvents()

    def on_embedding_finished(self, results):
        # Stop the progress timer
        self.progress_timer.stop()

        # Close progress dialog
        self.progress_dialog.close()

        # Enable main window
        self.centralWidget().setEnabled(True)

        # Display t-SNE plot
        self.plot_tsne(results["features"], results["labels"])

        # Display UMAP plot
        self.plot_umap(results["features"], results["labels"])

        # Display similarity search results
        self.display_similarity_results(results["similarity_results"])

        # Show completion message
        QMessageBox.information(self, "Analysis Complete", "Dataset embedding analysis is complete.")

    def plot_tsne(self, features, labels):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features)
        self.figure_tsne.clear()
        ax = self.figure_tsne.add_subplot(111)
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
        ax.set_title('t-SNE Visualization of Image Embeddings')
        self.canvas_tsne.draw()

    def plot_umap(self, features, labels):
        reducer = umap.UMAP()
        umap_result = reducer.fit_transform(features)
        self.figure_umap.clear()
        ax = self.figure_umap.add_subplot(111)
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
        ax.set_title('UMAP Visualization of Image Embeddings')
        self.canvas_umap.draw()

    def display_similarity_results(self, similarity_results):
        for i in reversed(range(self.similarity_layout.count())):
            widget_to_remove = self.similarity_layout.itemAt(i).widget()
            self.similarity_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        for image_path in similarity_results:
            pixmap = QPixmap(image_path)
            label = QLabel()
            label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
            self.similarity_layout.addWidget(label)

def main():
    app = QApplication(sys.argv)
    window = EmbeddingApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()