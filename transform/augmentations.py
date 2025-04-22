import albumentations as A 
import os
from torchvision import transforms, datasets
import cv2

def augmentationFolder(dataset_path, save_path, transform_method, number):
    dataset = datasets.ImageFolder(dataset_path)
    saved_path = f"{save_path}/AugmentedDataset"
    
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

        # Save the original image
        original_image = cv2.imread(path)
        original_image_path = os.path.join(saved_path, dataset.classes[cls_idx], f"{filename}_original{ext}").replace("\\", "/")
        cv2.imwrite(original_image_path, original_image)

        # Save augmented images
        for n in range(1, number + 1):
            image = cv2.imread(path)
            transformed = transform_method(image=image)
            transformed_image = transformed["image"]
            image_name = f"{filename}_{n}{ext}"
            image_path = os.path.join(saved_path, dataset.classes[cls_idx], image_name).replace("\\", "/")
            cv2.imwrite(image_path, transformed_image)