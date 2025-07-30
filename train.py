# === IMPORTS ===
import yaml  # For reading configuration files
from tqdm import tqdm  # For progress bars
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import copy

from model import BetterNet  # Your custom CNN architecture

# === CONFIGURATION LOADING ===
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# === DATA TRANSFORMS (AUGMENTATION + NORMALIZATION) ===
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomCrop(32, padding=4),  # Augmentation
    transforms.ToTensor(),  # Converts image to tensor [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization to [-1, 1]
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === DATALOADER FUNCTION ===
def get_loaders():
    # Ensure folders exist
    os.makedirs(config['paths']['train_dir'], exist_ok=True)
    os.makedirs(config['paths']['test_dir'], exist_ok=True)

    # Load datasets
    dataset_train_full = datasets.CIFAR10(root=config['paths']['train_dir'], train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10(root=config['paths']['test_dir'], train=False, download=True, transform=transform_test)

    # Split training set into train/val
    val_split = config['hyperparameters'].get('val_split', 0.1)
    n_total = len(dataset_train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    dataset_train, dataset_val = random_split(dataset_train_full, [n_train, n_val])

    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=config['hyperparameters']['batch_size'], shuffle=True, num_workers=config['hyperparameters']['num_workers'])
    dataloader_val = DataLoader(dataset_val, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=config['hyperparameters']['num_workers'])
    dataloader_test = DataLoader(dataset_test, batch_size=config['hyperparameters']['batch_size'], shuffle=False, num_workers=config['hyperparameters']['num_workers'])

    return dataloader_train, dataloader_val, dataloader_test

# === EVALUATION FUNCTION (USED IN VALIDATION + TEST) ===
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total, 100 * correct / total

# === TRAINING FUNCTION WITH EARLY STOPPING AND LR SCHEDULING ===
def train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device):
    epochs = config['hyperparameters']['epochs']
    patience = config['hyperparameters'].get('early_stopping_patience', 5)
    model_path = config['paths']['model_path']

    best_val_acc = 0.0
    epochs_since_improvement = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training loop with progress bar
        with tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})

        # Step the learning rate scheduler
        if scheduler:
            scheduler.step()

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        print(f"Epoch {epoch+1} | Train loss: {running_loss / len(dataloader_train):.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
            torch.save(best_model_state, model_path)
            print(f"New best model saved at {model_path}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s)")

        # Early stopping
        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            break

    # Restore best weights
    model.load_state_dict(best_model_state)
    print("✅ Restored best model state.")

# === FINAL TEST FUNCTION ===
def test(model, dataloader_test, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Final test accuracy: {100 * correct / total:.2f} %')

# === MAIN EXECUTION BLOCK ===
def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load everything
    dataloader_train, dataloader_val, dataloader_test = get_loaders()
    model = BetterNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['hyperparameters']['lr'], weight_decay=config['hyperparameters']['weight_decay'])

    # Learning rate scheduler: reduce LR every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Train model
    train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device)

    # Final evaluation
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')

    # Save final model
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_path'])
    print(f"Model saved to {config['paths']['model_path']}")

# Entry point
if __name__ == '__main__':
    main()
