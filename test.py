import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.getData import Data  
from sklearn.metrics import roc_curve, auc

def evaluate_model(model, data_loader, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.permute(0, 3, 1, 2).float() 
            trg = torch.argmax(trg, dim=1)  

            outputs = model(src)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(trg.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    all_labels_onehot = np.eye(num_classes)[all_labels]
    auc_score = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc_score, cm, all_labels_onehot, all_probs


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(ground_truth_bin, probabilities, class_names, save_path="roc_auc.png"):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(ground_truth_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def main():
    BATCH_SIZE = 4
    NUM_CLASSES = 6
    DATASET_PATH = {
        "aug": "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/",
        "orig": "./Dataset/Original Images/Original Images/FOLDS/",
    }
    MODEL_PATH = "trained_modelMobileNetV3Large.pth"

    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]

    dataset = Data(base_folder_aug=DATASET_PATH["aug"], base_folder_orig=DATASET_PATH["orig"])
    test_data = dataset.dataset_test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    metrics = evaluate_model(model, test_loader, NUM_CLASSES)
    accuracy, precision, recall, f1, auc_score, cm, all_labels_onehot, all_probs = metrics

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc_score:.4f}")
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png")
    plot_roc_curve(all_labels_onehot, all_probs, class_names, save_path="roc_auc.png")


if __name__ == "__main__":
    main()
