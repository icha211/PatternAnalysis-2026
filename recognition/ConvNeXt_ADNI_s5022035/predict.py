import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_adni_dataloaders
from modules import BaselineCNN, ConvNeXtTiny


def evaluate_and_autopsy(model, test_loader, device, save_dir="./results", confidence_threshold=0.7):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    failure_cases = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                prob_ad = probs[i][1].item()
                pred = preds[i].item()
                label = labels[i].item()

                all_probs.append(prob_ad)
                all_preds.append(pred)
                all_labels.append(label)

                if pred != label and len(failure_cases) < 5:
                    failure_cases.append({
                        'image': images[i].cpu().squeeze().numpy(),
                        'true_label': 'AD' if label == 1 else 'NC',
                        'pred_label': 'AD' if pred == 1 else 'NC',
                        'confidence': prob_ad if pred == 1 else (1.0 - prob_ad)
                    })

    print("\n================ Classification Report ================")
    print(classification_report(all_labels, all_preds, target_names=['NC', 'AD'], digits=4))
    
    auroc = roc_auc_score(all_labels, all_probs)
    print(f"Test AUROC Score: {auroc:.4f}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NC', 'AD'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("ADNI Test Set Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()

    if failure_cases:
        fig, axes = plt.subplots(1, len(failure_cases), figsize=(15, 3))
        if len(failure_cases) == 1:
            axes = [axes]

        for idx, case in enumerate(failure_cases):
            axes[idx].imshow(case['image'], cmap='gray')
            axes[idx].set_title(f"True: {case['true_label']} | Pred: {case['pred_label']}\nConf: {case['confidence']:.2f}")
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "failure_case_autopsy.png"))
        plt.close()
        print(f"Saved failure case autopsy plot to {os.path.join(save_dir, 'failure_case_autopsy.png')}")

    ambiguous_count = 0
    automated_correct = 0
    automated_total = 0

    for prob, pred, label in zip(all_probs, all_preds, all_labels):
        if 0.35 <= prob <= 0.65:
            ambiguous_count += 1
        else:
            automated_total += 1
            if pred == label:
                automated_correct += 1

    auto_acc = automated_correct / automated_total if automated_total > 0 else 0
    print(f"\n================ Operational Triage Policy Simulation ================")
    print(f"Total Test Cases     : {len(all_labels)}")
    print(f"Automated Cases      : {automated_total} ({automated_total / len(all_labels) * 100:.1f}%)")
    print(f"Routed to Human      : {ambiguous_count} ({ambiguous_count / len(all_labels) * 100:.1f}%)")
    print(f"Automated Sub-System Accuracy: {auto_acc * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Inference and Evaluation Script")
    parser.add_argument('--model', type=str, default='convnext', choices=['baseline', 'convnext'])
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_convnext_model.pt')
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_adni_dataloaders(base_dir=args.data_dir, batch_size=32)

    if args.model == 'baseline':
        model = BaselineCNN(in_channels=1, num_classes=2).to(device)
    else:
        model = ConvNeXtTiny(in_channels=1, num_classes=2).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded weights from {args.checkpoint}")

    evaluate_and_autopsy(model, test_loader, device)


if __name__ == '__main__':
    main()