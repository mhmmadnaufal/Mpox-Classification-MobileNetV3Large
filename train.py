import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import models
from Utils.getData import Data

def main():
    BATCH_SIZE = 32
    EPOCH = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6

    aug_path = "C:/IPSD-Assesment/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/IPSD-Assesment/Dataset/Original Images/Original Images/FOLDS/"

    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    full_data = dataset.dataset_train + dataset.dataset_aug

    train_size = int(0.8 * len(full_data))
    valid_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCH):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.permute(0, 3, 1, 2).float()
            labels = torch.argmax(labels, dim=1)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.permute(0, 3, 1, 2).float()
                labels = torch.argmax(labels, dim=1)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(valid_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}] "
              f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "trained_modelMobileNetV3Large.pth")

    plt.plot(range(EPOCH), train_losses, label='Train Loss', color='blue')
    plt.plot(range(EPOCH), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("training.png")
    plt.show()

if __name__ == "__main__":
    main()
