import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_adni_dataloaders
from modules import BaselineCNN, ConvNeXtTiny, build_torchvision_convnext_tiny


def aggregate_patient_predictions(patient_ids, probabilities, labels):
    if not (len(patient_ids) == len(probabilities) == len(labels)):
        raise ValueError("Patient IDs, probabilities, and labels must have equal lengths")

    patient_probabilities = {}
    patient_labels = {}
    for patient_id, probability, label in zip(patient_ids, probabilities, labels):
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient {patient_id!r} has inconsistent labels")
        patient_labels[patient_id] = label
        patient_probabilities.setdefault(patient_id, []).append(probability)

    ordered_ids = sorted(patient_probabilities)
    return (
        ordered_ids,
        [patient_labels[patient_id] for patient_id in ordered_ids],
        [
            float(np.mean(patient_probabilities[patient_id]))
            for patient_id in ordered_ids
        ],
    )


def evaluate_and_autopsy(model, test_loader, device, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    failure_cases = {"false_positive": [], "false_negative": []}
    patient_ids = test_loader.dataset.patient_ids
    image_offset = 0

    with torch.no_grad():
        for images, labels in test_loader:
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

                if pred != label:
                    error_type = "false_positive" if pred == 1 else "false_negative"
                    case_patient_id = patient_ids[image_offset + i]
                    if not any(
                        case["patient_id"] == case_patient_id
                        for case in failure_cases[error_type]
                    ):
                        failure_cases[error_type].append({
                            'patient_id': case_patient_id,
                            'image': images[i].cpu().squeeze().numpy(),
                            'true_label': 'AD' if label == 1 else 'NC',
                            'pred_label': 'AD' if pred == 1 else 'NC',
                            'confidence': prob_ad if pred == 1 else (1.0 - prob_ad),
                            'error_type': error_type,
                        })
                        failure_cases[error_type].sort(
                            key=lambda case: case["confidence"], reverse=True
                        )
                        failure_cases[error_type] = failure_cases[error_type][:3]
            image_offset += images.size(0)

    print("\n================ Classification Report ================")
    print(classification_report(all_labels, all_preds, target_names=['NC', 'AD'], digits=4))
    
    auroc = roc_auc_score(all_labels, all_probs)
    print(f"Test AUROC Score: {auroc:.4f}")

    patient_ids, patient_labels, patient_probs = aggregate_patient_predictions(
        patient_ids, all_probs, all_labels
    )
    patient_preds = [int(probability >= 0.5) for probability in patient_probs]
    print("\n================ Patient-Level Metrics (mean slice probability) ================")
    print(
        classification_report(
            patient_labels,
            patient_preds,
            labels=[0, 1],
            target_names=['NC', 'AD'],
            digits=4,
            zero_division=0,
        )
    )
    print(f"Patient-level AUROC: {roc_auc_score(patient_labels, patient_probs):.4f}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NC', 'AD'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("ADNI Test Set Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()

    selected_failure_cases = (
        failure_cases["false_positive"] + failure_cases["false_negative"]
    )
    if selected_failure_cases:
        fig, axes = plt.subplots(
            1, len(selected_failure_cases),
            figsize=(4 * len(selected_failure_cases), 3),
            squeeze=False,
        )

        for idx, case in enumerate(selected_failure_cases):
            axes[0][idx].imshow(case['image'], cmap='gray')
            axes[0][idx].set_title(
                f"{case['error_type'].replace('_', ' ').title()}\n"
                f"Patient: {case['patient_id']} | Conf: {case['confidence']:.2f}"
            )
            axes[0][idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "failure_case_autopsy.png"))
        plt.close()
        print(
            f"Saved up to three high-confidence false positives and false negatives "
            f"to {os.path.join(save_dir, 'failure_case_autopsy.png')}"
        )

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
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for test plots and failure-case images')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load the ImageNet-initialized ConvNeXt model architecture')
    args = parser.parse_args()

    if args.pretrained and args.model != 'convnext':
        parser.error('--pretrained is only supported with --model convnext')

    if args.checkpoint is None:
        if args.model == 'baseline':
            args.checkpoint = './checkpoints/best_baseline_model.pt'
        elif args.pretrained:
            args.checkpoint = './checkpoints/best_convnext_pretrained_model.pt'
        else:
            args.checkpoint = './checkpoints/best_convnext_model.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_adni_dataloaders(base_dir=args.data_dir, batch_size=32)

    if args.model == 'baseline':
        model = BaselineCNN(in_channels=1, num_classes=2).to(device)
    elif args.pretrained:
        model = build_torchvision_convnext_tiny(pretrained=False, num_classes=2).to(device)
    else:
        model = ConvNeXtTiny(in_channels=1, num_classes=2).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded weights from {args.checkpoint}")

    evaluate_and_autopsy(model, test_loader, device, save_dir=args.results_dir)


if __name__ == '__main__':
    main()