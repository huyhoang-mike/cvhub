import albumentations as A 
import os
from torchvision import transforms, datasets
import cv2

def preprocessedFolder(dataset_path, save_path, transform_method):
    dataset = datasets.ImageFolder(dataset_path)
    saved_path = f"{save_path}/PreprocessedDataset"
    
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    
    for cls in dataset.classes:
        class_path = os.path.join(saved_path, cls)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
    
    for i in dataset.imgs:
        path = i[0]
        cls_idx = i[1]
        basename = os.path.basename(path)
        split = os.path.splitext(basename)
        filename = split[0]
        ext = split[1]

        image = cv2.imread(path)

        if transform_method is not None:
            for step in transform_method:
                transformed = step(image=image)
                image = transformed["image"]
    
        image_name = f"{filename}_preprocessed{ext}"
        image_path = os.path.join(saved_path, dataset.classes[cls_idx], image_name).replace("\\","/")
        cv2.imwrite(image_path, image)