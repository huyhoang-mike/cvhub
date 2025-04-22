from templates.pytorch.generate import generate_script
import albumentations as A

augmentation = A.Compose([
    A.RandomCrop(width=128, height=128),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=1),
    A.GaussNoise(var_limit=(10,50),p=1.0)
])

preprocess = A.Compose([
    A.Resize(width=128, height=128),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

with open("./templates/classification.py", "r") as template_file:
    template_text = template_file.read()

script = generate_script(
    template_content=template_text,
    dataset_dir="",
    split_ratio=[0.7,0.2,0.1],
    pretrained="",
    epochs=20,
    batch_size=32,
    learning_rate=0.0001,
    device="cpu",
    augmentation_pipeline=augmentation,
    preprocessing_pipeline=preprocess
)

print(script)