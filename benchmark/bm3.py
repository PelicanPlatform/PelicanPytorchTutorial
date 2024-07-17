import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models

import fsspec
from pelicanfs.core import PelicanFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.cached import WholeFileCacheFileSystem

from PIL import Image
import warnings
warnings.filterwarnings("ignore")
mp.set_start_method('spawn', force=True)

# Local datas path
local_trainfile_path = "ImageNetMini/train"
local_valfile_path = "ImageNetMini/val"

# Define the Pelican paths
trainfile_path = "/chtc/PUBLIC/hzhao292/ILSVRC/Data/CLS-LOC/train"
valfile_path = "/chtc/PUBLIC/hzhao292/ILSVRC/Data/CLS-LOC/val"

dev_trainfile_path = "/chtc/PUBLIC/hzhao292/ImageNetMini/train"
dev_valfile_path = "/chtc/PUBLIC/hzhao292/ImageNetMini/val"

# Define the transformer.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Ensure ToTensor is included
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Ensure ToTensor is included
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-e', '--epochs', default=5, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet50', type=str, metavar='N',
                    help='model architecture (default: resnet50)')

class RemoteImageFolder(VisionDataset):

    def __init__(self, root, fs = LocalFileSystem, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.fs = fs
        if os.path.isdir(root):
            self._init_local(root)
        else:
            self._init_remote(root)

    def _init_local(self, root):
        print(f"Initializing local dataset from {root}")
        self.root = root
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.imgs = self._make_dataset_local()

    def _init_remote(self, root, transform=None):
        self.root = root
        self.classes = sorted([item['name'] for item in self.fs.ls(root)])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.imgs = self._make_dataset_remote()

    def _make_dataset_local(self):
        images = []
        for class_idx, cls_name in enumerate(self.classes):
            class_path = os.path.join(self.root, cls_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_name = img_name.lower()
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    images.append((img_path, class_idx))
        return images

    def _make_dataset_remote(self):
        images = []
        for class_idx, cls_name in enumerate(self.classes):
            class_path = os.path.join(self.root, cls_name)
            files = self.fs.ls(class_path)
            for item in files:
                img_path = item['name']
                if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') or img_path.lower().endswith('.png'):
                    images.append((img_path, class_idx))
        print("len(images): ", len(images))
        return images

    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        if isinstance(self.fs, PelicanFileSystem) or isinstance(self.fs, WholeFileCacheFileSystem): 
            with self.fs.open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        else:
            print("img_path: ", img_path)
            img = read_image(img_path)
            img = transforms.ToPILImage()(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.imgs)



def training(train_dataset, train_loader, val_loader):
    args = parser.parse_args()

    # Initialize the model, loss function, and optimizer
    if args.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.arch=='vgg16':
        model = models.vgg16(pretrained=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and validation loop
    num_epochs = args.epochs

    print("Training started.")
    for epoch in range(num_epochs):
        start_time = time.time()
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader.dataset)
        accuracy = correct / total
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_loss:.2f}, Accuracy: {accuracy:.2f}, Time Taken: {time_taken:.2f} seconds")
    print("Training completed.")


def check_files_in_folder(folder_path):
    try:
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Check if there are any files
        if not files:
            print("No files found in the folder.")
            return

        # Print the sizes of the files
        for file in files:
            file_path = os.path.join(folder_path, file)
            file_size = os.path.getsize(file_path)
            print(f"File: {file}, Size: {file_size} bytes")

    except Exception as e:
        print(f"An error occurred: {e}")

def train_locally():
    args = parser.parse_args()
    print()
    print("Read Locally.")

    start_time = time.time()

    train_dataset = RemoteImageFolder(root=local_trainfile_path, transform=train_transforms)
    val_dataset = RemoteImageFolder(root=local_valfile_path, transform=val_transforms)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    end_time = time.time()
    print(f"Data preparing time: {end_time-start_time:4f}.")
    training(train_dataset, train_loader, val_loader)


def train_remote():
    # # Read data remotely from Pelican
    print()
    print("Read data remotely from Pelican")
    args = parser.parse_args()

    start_time = time.time()
    
    fs = PelicanFileSystem("pelican://osg-htc.org")

    # Load the datasets
    train_dataset = RemoteImageFolder(root=dev_trainfile_path,fs=fs,transform=train_transforms)
    val_dataset = RemoteImageFolder(root=dev_valfile_path,fs=fs,transform=val_transforms)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    end_time = time.time()
    print(f"Data preparing time: {end_time-start_time:4f}.")

    training(train_dataset, train_loader, val_loader)


def train_remote_localcache():
    args = parser.parse_args()
    # ## Read data remotely from Pelican with local Cache
    print()
    print("Read data remotely from Pelican with local Cache")

    start_time = time.time()
    # Load the datasets
    fs = fsspec.filesystem("filecache", target_protocol='osdf', cache_storage='tmp/files/')
    train_dataset = RemoteImageFolder(root=dev_trainfile_path, fs=fs, transform=train_transforms)
    val_dataset = RemoteImageFolder(root=dev_valfile_path, fs=fs, transform=val_transforms)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    end_time = time.time()
    print(f"Data preparing time: {end_time-start_time:4f}.")

    training(train_dataset, train_loader, val_loader)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_locally()
    train_remote()
    train_remote_localcache()