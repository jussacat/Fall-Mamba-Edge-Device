import torch
from torchvision import transforms

def get_transforms(is_train=True):
    base_transforms = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), #Resnet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
    ]

    return transforms.Compose(base_transforms)