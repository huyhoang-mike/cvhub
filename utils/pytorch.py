import torch
import platform
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

def get_all_available_devices():
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.cuda.get_device_properties(i).name)
    devices.append(platform.processor())
    return devices

def getDevice(selected_device):
    if selected_device == platform.processor():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    return device

