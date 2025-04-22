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


class CustomImageFolder(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # Get the original tuple
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return path, sample, target

class classificationModel:
    def __init__(
            self,
            dataset_dir,
            split_ratio, 
            backbone, 
            device, 
            epochs, 
            batch_size, 
            lr, 
            loss_fn, 
            optimizer
    ):

        self.device = device
        self.epochs = epochs

        mainDataset = self.getDataset(directory=dataset_dir)
        self.classes = mainDataset.classes
        train_dataset, self.valid_dataset, self.test_dataset = self.getSplit(dataset=mainDataset, split_ratio=split_ratio)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.loss_fn = self.getLossfn(literal=loss_fn)
        self.getBackbone(literal=backbone, num_classes=len(self.classes))
        self.optimizer = self.getOptimizer(literal=optimizer, model=self.model, lr=lr)

        self.datsetLength = self.getLength(dataset=mainDataset, split_ratio=split_ratio)

    def getOptimizer(self, model, lr, literal):
        if literal == "RMSCrop":
            optimizer = optim.RMSprop(params=model.parameters(), lr=lr)
        elif literal == "Adam":
            optimizer = optim.Adam(params=model.parameters(), lr=lr)
        elif literal == "AdamW":
            optimizer = optim.AdamW(params=model.parameters(), lr=lr)
        elif literal == "SGD":
            optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
        elif literal == "Adagrad":
            optimizer = optim.Adagrad(params=model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
        return optimizer

    def getLossfn(self, literal):
        if literal == "Cross Entropy Loss":
            loss_fn = nn.CrossEntropyLoss()
        elif literal == "Binary Cross Entropy Loss":
            loss_fn = nn.BCELoss()
        elif literal == "Cosine Embedding Loss":
            loss_fn = nn.CosineEmbeddingLoss()
        elif literal == "Hinge Embedding Loss":
            loss_fn = nn.HingeEmbeddingLoss()
        elif literal == "Negative Log Likelihood Loss":
            loss_fn = nn.NLLLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def getBackbone(self, literal, num_classes):
        if literal == "Resnet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)
        elif literal == "GoogleNet":
            self.model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)
        elif literal == 'Vgg19':
            self.model = vgg19(VGG19_Weights.IMAGENET1K_V1)
            self.modifyModel(model=self.model, num_classes=num_classes)
        elif literal == 'AlexNet':
            self.model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            self.modifyModel(model=self.model, num_classes=num_classes)
        elif literal == 'Efficienetb0':
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.modifyModel(model=self.model, num_classes=num_classes)  
    
    def modifyModel(self, model, num_classes):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    def getDataset(self, directory):

        torch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = CustomImageFolder(directory, transform=torch_transform)
        return dataset

    def getSplit(self, dataset, split_ratio):
        train_size = int(split_ratio[0] * len(dataset))  
        valid_size = int(split_ratio[1] * len(dataset))  
        test_size = len(dataset) - train_size - valid_size  

        # Split the dataset
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

        return train_dataset, valid_dataset, test_dataset

    def getLength(self, dataset, split_ratio):
        train_size = int(split_ratio[0] * len(dataset))  
        valid_size = int(split_ratio[1] * len(dataset))  
        test_size = len(dataset) - train_size - valid_size
        return [train_size, valid_size, test_size]

class classificationTrainThread(QThread):

    loss_signal = pyqtSignal(float)
    error_signal = pyqtSignal(float)
    time_elapsed_signal = pyqtSignal(float)
    time_left_signal = pyqtSignal(float)
    progress_bar = pyqtSignal(int)

    def __init__(
            self,
            model,
            device,
            epochs,
            train_loader,
            criterion,
            optimizer
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    @pyqtSlot()
    def run(self):
        self.model.to(self.device)

        total_start_time = time.time()  # Start time for the entire training process
        time_left_previous = 1e6

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            self.model.train()

            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for paths, inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    pbar.update(1)

            self.exp_lr_scheduler.step()

            elapsed_time = time.time() - total_start_time  # Total elapsed time since the start of training
            fraction_completed = (epoch + 1) / self.epochs
            estimated_total_time = elapsed_time / fraction_completed
            time_left = estimated_total_time - elapsed_time  # Estimated time left for the entire training process

            self.time_elapsed_signal.emit(elapsed_time)

            if time_left < time_left_previous:       
                time_left_previous = time_left 
                self.time_left_signal.emit(time_left)

            average_loss = epoch_loss / len(self.train_loader)
            top1_error = 1 - correct / total

            # Emit signals with the loss and error rate
            self.loss_signal.emit(average_loss)
            self.error_signal.emit(top1_error)

            if time_left != 0:
                self.progress_bar.emit(int((elapsed_time/time_left)*100))

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {average_loss:.4f}, Top1 Error Rate: {top1_error:.4f}')

        total_elapsed_time = time.time() - total_start_time
        self.progress_bar.emit(100)
        print(f'Total training time: {total_elapsed_time:.2f} seconds')

class classificationEvalThread(QThread):

    f1 = pyqtSignal(float)
    precision = pyqtSignal(float)
    recall = pyqtSignal(float)
    total_images = pyqtSignal(int)
    confusion_matrix = pyqtSignal(object)

    def __init__(
            self,
            model,
            device,
            valid_data,
            batch_size_eval
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.valid_loader = DataLoader(valid_data, batch_size=batch_size_eval, shuffle=False)

    @pyqtSlot()
    def run(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_images = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                paths, inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                    
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_images += inputs.size(0)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        self.f1.emit(f1)
        self.precision.emit(precision)
        self.recall.emit(recall)
        self.total_images.emit(total_images)
        self.confusion_matrix.emit(conf_matrix)

class classificationInferenceThread(QThread):

    inference_finish = pyqtSignal(object, object, object, object)

    def __init__(self, model, device, test_data, batch_size_inference):
        super().__init__()
        self.model = model
        self.device = device
        self.test_loader = DataLoader(test_data, batch_size=batch_size_inference, shuffle=False)

    @pyqtSlot()
    def run(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_image_paths = []
        all_confidences = []

        with torch.no_grad():
            for batch in self.test_loader:
                paths, inputs, labels = batch
                all_image_paths.extend(paths)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Get the predicted class and corresponding confidence score
                confidence_scores, preds = torch.max(probabilities, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence_scores.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)

        self.inference_finish.emit(all_image_paths, all_labels, all_preds, all_confidences)


class classificationInferenceThread_test(QThread):

    inference_finished = pyqtSignal(object)

    def __init__(self, model, input_data, device=torch.device("cpu"), parent=None):
        super(classificationInferenceThread_test, self).__init__(parent)
        self.model = model.to(device)
        self.device = device
        self.input_data = input_data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    @pyqtSlot()
    def run(self):
        result = self.infer(self.input_data)
        self.inference_finished.emit(result)

    def infer_dataloader(self, dataloader: DataLoader):
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in dataloader:
                paths, inputs, labels = batch

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Get the predicted class and corresponding confidence score
                confidences, predictions = torch.max(probabilities, 1)

                for i in range(len(inputs)):
                    result = {
                        "image_path": paths[i],
                        "label": labels[i].cpu().numpy().tolist(),
                        "prediction": predictions[i].cpu().numpy().tolist(),
                        "confidence": confidences[i].cpu().numpy().tolist()
                    }
                    results.append(result)
        return results

    def infer_image_folder(self, folder_path: str):
        images = []
        filenames = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                images.append(img)
                filenames.append(img_path)

        images_tensor = torch.stack(images).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images_tensor)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the predicted class and corresponding confidence score
            confidences, predictions = torch.max(probabilities, 1)
        
        results = []
        for i, filename in enumerate(filenames):
            result = {
                "image_path": filename,
                "label": None,  # No labels in the image folder
                "prediction": predictions[i].cpu().numpy().tolist(),
                "confidence": confidences[i].cpu().numpy().tolist()
            }
            results.append(result)
        return results

    def infer_single_image(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

        result = {
            "image_path": image_path,
            "label": None,  # No label for a single image
            "prediction": predictions[0].cpu().numpy().tolist(),
            "confidence": confidences[0].cpu().numpy().tolist()
        }
        return result

    def infer(self, input_data):
        if isinstance(input_data, DataLoader):
            return self.infer_dataloader(input_data)
        elif isinstance(input_data, str):
            if os.path.isdir(input_data):
                return self.infer_image_folder(input_data)
            elif os.path.isfile(input_data):
                return self.infer_single_image(input_data)
            else:
                raise ValueError("Provided string is neither a file nor a directory.")
        else:
            raise TypeError("Input data must be a DataLoader, directory path, or image file path.")