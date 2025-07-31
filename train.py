import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import os

from model import SimpleNet

# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Standard input normalization for CIFAR-10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# These transformations augment images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# These don't (validation/testing)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def get_loaders():
    # Ensure train and test data directories exist
    os.makedirs(config['paths']['train_dir'], exist_ok=True)
    os.makedirs(config['paths']['test_dir'], exist_ok=True)

    # Load the full training set
    dataset_train_full = datasets.CIFAR10(
        root=config['paths']['train_dir'],
        train=True,
        download=True,
        transform=train_transform
    )
    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'],
        train=False,
        download=True,
        transform=test_transform
    )

    # Split train into train/val (e.g., 90% train, 10% val)
    val_split = config['hyperparameters'].get('val_split', 0.1)
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    dataset_train, dataset_val = random_split(dataset_train_full, [n_train, n_val])

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )
    return dataloader_train, dataloader_val, dataloader_test


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

train_losses = []
val_losses = []
val_accuracies = []

def train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device):
    model.train()
    epochs = config['hyperparameters']['epochs']
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(
            dataloader_train,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            unit="batch"
        ) as progress_bar:
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': avg_loss})
        avg_train_loss = running_loss / len(dataloader_train)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")
        scheduler.step()
    print('Finished Training')


# Function to test the neural network and print accuracy
def test(model, dataloader_test, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Count of correct predictions
    total = 0  # Total number of samples
    with torch.no_grad():  # No need to compute gradients during testing (save memory)
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Update total count
            correct += (predicted == labels).sum().item()  # Update correct count
    # Print accuracy as a percentage
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')


def main():
    # Select device: use GPU if available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders for training and testing
    dataloader_train, dataloader_val, dataloader_test = get_loaders()

    # Create the neural network and move it to the selected device
    model = SimpleNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparameters']['lr'], weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.5)

    train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device)
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')

    # Save the trained model's parameters to a file
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)  # Create weights dir if missing
    torch.save(model.state_dict(), config['paths']['model_path']) # Save the model's parameters to a pth file
    print(f"Model saved to {config['paths']['model_path']}")

    epochs_range = range(1, config['hyperparameters']['epochs'] + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label="Val Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == '__main__':
    main()
