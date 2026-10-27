import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class ADNIDataset(Dataset):
    def __init__(self, file_paths, labels, patient_ids, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.patient_ids = patient_ids
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
    stem, _ = os.path.splitext(filename)
    parts = stem.split('_')
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Expected an image filename in patient_slice format, got {filename!r}"
        )
    return parts[0]


def load_split_files(split_dir):
    class_map = {"NC": 0, "AD": 1}  
    patient_files = {}
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Expected class directory does not exist: {class_dir}")
        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(class_dir, filename)
                patient_id = extract_patient_id(filename)
                if patient_id not in patient_files:
                    patient_files[patient_id] = {"label": label, "files": []}
                elif patient_files[patient_id]["label"] != label:
                    raise ValueError(
                        f"Patient ID {patient_id!r} has files in both classes under {split_dir}. "
                        "Resolve longitudinal or conflicting labels before training."
                    )
                patient_files[patient_id]["files"].append((file_path, label))
    if not patient_files:
        raise ValueError(f"No supported images found under {split_dir}")
    return patient_files


def seed_data_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_adni_dataloaders(
    base_dir="/home/groups/comp3710/ADNI/AD_NC",
    batch_size=32,
    img_size=224,
    val_ratio=0.15,
    random_state=42,
    num_workers=1,
):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_patient_files = load_split_files(train_dir)
    test_patient_files = load_split_files(test_dir)

    overlapping_patients = set(train_patient_files).intersection(test_patient_files)
    if overlapping_patients:
        raise ValueError(
            f"Patient-level leakage detected: {len(overlapping_patients)} patient IDs "
            f"occur in both train and test (examples: "
            f"{', '.join(sorted(overlapping_patients)[:5])})."
        )
    
    train_patient_ids = list(train_patient_files.keys())
    train_labels = [train_patient_files[patient_id]["label"] for patient_id in train_patient_ids]
    
    tr_patient_ids, val_patient_ids = train_test_split(
        train_patient_ids,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_labels
    )
    
    def assemble_paths(patient_files, patient_ids):
        paths, labels, assembled_patient_ids = [], [], []
        for patient_id in patient_ids:
            for path, label in patient_files[patient_id]["files"]:
                paths.append(path)
                labels.append(label)
                assembled_patient_ids.append(patient_id)
        return paths, labels, assembled_patient_ids

    tr_paths, tr_labels, tr_patient_ids = assemble_paths(
        train_patient_files, tr_patient_ids
    )
    val_paths, val_labels, val_patient_ids = assemble_paths(
        train_patient_files, val_patient_ids
    )
    test_patient_ids = list(test_patient_files.keys())
    test_paths, test_labels, test_patient_ids = assemble_paths(
        test_patient_files, test_patient_ids
    )

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ADNIDataset(
        tr_paths, tr_labels, tr_patient_ids, transform=train_transform
    )
    val_dataset = ADNIDataset(
        val_paths, val_labels, val_patient_ids, transform=eval_transform
    )
    test_dataset = ADNIDataset(
        test_paths, test_labels, test_patient_ids, transform=eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_data_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader