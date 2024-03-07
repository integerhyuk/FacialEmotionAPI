import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample



def stack_to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def normalize_tensor(tensors):
    mu, st = 0, 255

    return torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])

def random_erasing(tensors):
    return torch.stack([transforms.RandomErasing()(t) for t in tensors])

def get_dataloaders(path='./AffectNet-8/valid', bs=64, augment=True):

    mu, st = 0, 255

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48,48)),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48,48)),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.FiveCrop(40),
            transforms.Lambda(stack_to_tensor),
            transforms.Lambda(normalize_tensor),
            transforms.Lambda(random_erasing),
        ])

    else:
        train_transform = test_transform

    train_dir = './RAF-DB/Resized'
    test_dir = './RAF-DB/Resized'

    train = ImageFolder(train_dir, transform=train_transform)
    test = ImageFolder(test_dir, transform=test_transform)


    trainloader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    #valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=32, shuffle=True, num_workers=0)

    return trainloader, testloader
