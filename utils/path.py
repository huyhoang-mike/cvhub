import os

def load_images_and_labels(data_dir):
    images_name = []
    labels = []
    paths = []
    classes = os.listdir(data_dir)

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if file_path.endswith(('.png', '.jpg', '.jpeg')):
                    images_name.append(file)
                    labels.append(class_name)
                    paths.append(file_path)
    
    return images_name, labels, paths