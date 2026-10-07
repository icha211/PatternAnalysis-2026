import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class ADNIDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("L")  # 1-channel grayscale MRI
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def extract_patient_id(file_path):
    filename = os.path.basename(file_path)
    return filename.split('_')[0]


def load_split_files(split_dir):
    class_map = {"NC": 0, "AD": 1}  
    patient_files = {}
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):  
                patient_id = extract_patient_id(filename)
                if patient_id not in patient_files:
                    patient_files[patient_id] = []
                patient_files[patient_id].append((os.path.join(class_dir, filename), label))
    return patient_files


def get_adni_dataloaders(base_dir="/home/groups/comp3710/ADNI/AD_NC", batch_size=32, img_size=224, val_ratio=0.15, random_state=42):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_patient_files = load_split_files(train_dir)
    test_patient_files = load_split_files(test_dir)
    
    train_pids = list(train_patient_files.keys())
    train_labels = [train_patient_files[pid][0][1] for pid in train_pids]
    
    tr_pids, val_pids = train_test_split(
        train_pids,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_labels
    )
    
    def assemble_paths(patient_files, pids):
        paths, labels = [], []
        for pid in pids:
            for path, label in patient_files[pid]:
                paths.append(path)
                labels.append(label)
        return paths, labels

    tr_paths, tr_labels = assemble_paths(train_patient_files, tr_pids)
    val_paths, val_labels = assemble_paths(train_patient_files, val_pids)
    test_pids = list(test_patient_files.keys())
    test_paths, test_labels = assemble_paths(test_patient_files, test_pids)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ADNIDataset(tr_paths, tr_labels, transform=train_transform)
    val_dataset = ADNIDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = ADNIDataset(test_paths, test_labels, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader