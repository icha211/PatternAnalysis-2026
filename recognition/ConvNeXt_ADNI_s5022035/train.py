import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

from dataset import get_adni_dataloaders, seed_data_worker
from modules import (
    CONVNEXT_CONFIGS,
    BaselineCNN,
    ConvNeXtTiny,
    build_torchvision_convnext_tiny,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation or test sets.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_metrics(history, save_path):
    """
    Plots and saves training & validation loss and accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_balanced_sample_weights(dataset):
    patient_labels = {}
    patient_counts = {}
    for patient_id, label in zip(dataset.patient_ids, dataset.labels):
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient {patient_id!r} has conflicting training labels")
        patient_labels[patient_id] = label
        patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1

    patients_per_class = {}
    for patient_id, label in patient_labels.items():
        patients_per_class.setdefault(label, set()).add(patient_id)

    weights = []
    for patient_id, label in zip(dataset.patient_ids, dataset.labels):
        class_patient_count = len(patients_per_class[label])
        weights.append(
            1.0 / (len(patients_per_class) * class_patient_count * patient_counts[patient_id])
        )
    return torch.as_tensor(weights, dtype=torch.double)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline or ConvNeXt model on ADNI dataset")
    parser.add_argument('--model', type=str, default='convnext', choices=['baseline', 'convnext'], help='Model architecture')
    parser.add_argument(
        '--convnext_variant',
        type=str,
        default='tiny',
        choices=sorted(CONVNEXT_CONFIGS),
        help='Custom ConvNeXt configuration to use for scratch training',
    )
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC', help='ADNI dataset directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='AdamW weight decay for regularization')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Cross-entropy label smoothing in [0, 1)')
    parser.add_argument('--patience', type=int, default=6,
                        help='Stop after this many epochs without validation-accuracy improvement; 0 disables')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader worker count')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                        help='Maximum stochastic-depth rate for scratch ConvNeXt')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help='Classifier dropout for scratch ConvNeXt')
    parser.add_argument('--layer_scale_init', type=float, default=1e-6,
                        help='Initial residual layer scale for scratch ConvNeXt')
    parser.add_argument(
        '--patient_balanced_sampling',
        action='store_true',
        help='Sample classes equally and patients equally within each class',
    )
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--pretrained', action='store_true',
                        help='Fine-tune torchvision ConvNeXt-Tiny initialized with ImageNet weights')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip test-set evaluation while comparing training experiments')
    parser.set_defaults(skip_test=True)
    parser.add_argument(
        '--evaluate_test',
        dest='skip_test',
        action='store_false',
        help='Explicitly evaluate the test set after training (prefer predict.py for final reporting)',
    )
    args = parser.parse_args()

    if args.pretrained and args.model != 'convnext':
        parser.error('--pretrained is only supported with --model convnext')
    if args.pretrained and args.convnext_variant != 'tiny':
        parser.error('--convnext_variant compact is only supported for scratch training')
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error('--label_smoothing must be in [0, 1)')
    if args.patience < 0:
        parser.error('--patience must be non-negative')
    if not 0.0 <= args.drop_path_rate < 1.0:
        parser.error('--drop_path_rate must be in [0, 1)')
    if not 0.0 <= args.head_dropout < 1.0:
        parser.error('--head_dropout must be in [0, 1)')
    if args.layer_scale_init < 0.0:
        parser.error('--layer_scale_init must be non-negative')
    if args.pretrained and (args.drop_path_rate or args.head_dropout):
        parser.error('--drop_path_rate and --head_dropout apply only to scratch ConvNeXt')

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_adni_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
        random_state=args.seed,
        num_workers=args.num_workers,
    )

    if args.model == 'baseline':
        model = BaselineCNN(in_channels=1, num_classes=2).to(device)
        checkpoint_name = "best_baseline_model.pt"
    else:
        if args.pretrained:
            model = build_torchvision_convnext_tiny(pretrained=True, num_classes=2).to(device)
            checkpoint_name = "best_convnext_pretrained_model.pt"
        else:
            model = ConvNeXtTiny(
                in_channels=1,
                num_classes=2,
                drop_path_rate=args.drop_path_rate,
                head_dropout=args.head_dropout,
                layer_scale_init=args.layer_scale_init,
                variant=args.convnext_variant,
            ).to(device)
            checkpoint_name = (
                "best_convnext_compact_model.pt"
                if args.convnext_variant == "compact"
                else "best_convnext_model.pt"
            )

    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    if args.patient_balanced_sampling:
        sampler = WeightedRandomSampler(
            patient_balanced_sample_weights(train_loader.dataset),
            num_samples=len(train_loader.dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            worker_init_fn=seed_data_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = float('-inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, eval_criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if args.patience and epochs_without_improvement >= args.patience:
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")

    best_checkpoint = os.path.join(args.save_dir, checkpoint_name)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    if not args.skip_test:
        test_loss, test_acc = evaluate(model, test_loader, eval_criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    curve_name = (
        f"convnext_{args.convnext_variant}_training_curves.png"
        if args.model == 'convnext' and not args.pretrained
        else f"{args.model}_training_curves.png"
    )
    if args.pretrained:
        curve_name = "convnext_pretrained_training_curves.png"
    plot_metrics(history, os.path.join(args.save_dir, curve_name))


if __name__ == '__main__':
    main()