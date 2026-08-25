import torch
from torchvision import transform

def get_transform(is_train=True):
    base_transforms = [
        transform.ToPILImage(),
        transform.Resize((224, 224)), #Resnet input size
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet
    ]

return transform.Compose(base_transforms)