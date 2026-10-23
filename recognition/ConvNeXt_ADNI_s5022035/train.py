import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_adni_dataloaders
from modules import BaselineCNN, ConvNeXtTiny, build_torchvision_convnext_tiny


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
    print(f"Metrics plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Baseline or ConvNeXt model on ADNI dataset")
    parser.add_argument('--model', type=str, default='convnext', choices=['baseline', 'convnext'], help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC', help='ADNI dataset directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--pretrained', action='store_true',
                        help='Fine-tune torchvision ConvNeXt-Tiny initialized with ImageNet weights')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip test-set evaluation while comparing training experiments')
    args = parser.parse_args()

    if args.pretrained and args.model != 'convnext':
        parser.error('--pretrained is only supported with --model convnext')

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load 
    train_loader, val_loader, test_loader = get_adni_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size
    )

    if args.model == 'baseline':
        model = BaselineCNN(in_channels=1, num_classes=2).to(device)
        model_name = "BaselineCNN"
        checkpoint_name = "best_baseline_model.pt"
    else:
        if args.pretrained:
            model = build_torchvision_convnext_tiny(pretrained=True, num_classes=2).to(device)
            model_name = "ImageNet-pretrained ConvNeXtTiny"
            checkpoint_name = "best_convnext_pretrained_model.pt"
        else:
            model = ConvNeXtTiny(in_channels=1, num_classes=2).to(device)
            model_name = "ConvNeXtTiny"
            checkpoint_name = "best_convnext_model.pt"

    print(f"Initialized {model_name} (Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")
    if args.pretrained:
        print("ImageNet weights are used for initialization; the full model is fine-tuned.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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
            checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")

    best_checkpoint = os.path.join(args.save_dir, checkpoint_name)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    if not args.skip_test:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"\n--- Held-Out Test Evaluation ({model_name}) ---")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    else:
        print("\nSkipping held-out test evaluation for this experiment.")

    curve_name = f"{args.model}_training_curves.png"
    if args.pretrained:
        curve_name = "convnext_pretrained_training_curves.png"
    plot_metrics(history, os.path.join(args.save_dir, curve_name))


if __name__ == '__main__':
    main()