import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QCheckBox, QFileDialog, QSpinBox, QDoubleSpinBox, QTextBrowser
)
from PyQt6.QtCore import Qt
from jinja2 import Template
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, Rotate
from albumentations.pytorch import ToTensorV2
import albumentations as A

def generate_script(
        template_content,
        dataset_dir,
        split_ratio,
        pretrained,
        epochs,
        batch_size,
        learning_rate,
        device,
        augmentation_pipeline,
        preprocessing_pipeline
    ):

    augmentation_pipeline_code = serialize_pipeline(augmentation_pipeline, False)
    # preprocessing_pipeline_code = serialize_pipeline(preprocessing_pipeline, True)

    template = Template(template_content)
    rendered_script = template.render(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        augmentation=augmentation_pipeline_code,
        preprocessing=preprocessing_pipeline
    )

    return rendered_script

def serialize_pipeline(pipeline, toTensor):
    """Serialize the Albumentations pipeline to Python code."""
    if toTensor:
        pipeline_code = "preprocessing_transform = A.Compose([\n"
    else:
        pipeline_code = "augmentation_transform = A.Compose([\n"
    
    for transform in pipeline.transforms:
        pipeline_code += f"    {repr(transform)},\n"

    if toTensor:
        pipeline_code += "    ToTensorV2()\n"

    pipeline_code += "])"
    return pipeline_code