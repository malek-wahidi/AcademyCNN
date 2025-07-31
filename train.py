import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from model import SimpleNet


# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Standard input normalization for CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_loaders():
    os.makedirs(config['paths']['train_dir'], exist_ok=True)
    os.makedirs(config['paths']['test_dir'], exist_ok=True)

    dataset_train_full = datasets.CIFAR10(
        root=config['paths']['train_dir'],
        train=True,
        download=True,
        transform=transform_train
    )

    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'],
        train=False,
        download=True,
        transform=transform_test
    )

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
        print(f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")
        scheduler.step()
        print(f"Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}")
    print('Finished Training')


def test(model, dataloader_test, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataloader_train, dataloader_val, dataloader_test = get_loaders()

    model = SimpleNet().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5) 
        
    train(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device)
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')

    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_path'])
    print(f"Model saved to {config['paths']['model_path']}")

if __name__ == '__main__':
    main()

