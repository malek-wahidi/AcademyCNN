import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

from model import SimpleNet

# Load config (YAML for easy editing)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Standard input normalization for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL image to PyTorch Tensor and normalize to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Mean and Standard Deviation of the CIFAR-10 dataset
])

def get_loaders():
    # Ensure train and test data directories exist
    os.makedirs(config['paths']['train_dir'], exist_ok=True)  # Create train data dir if missing
    os.makedirs(config['paths']['test_dir'], exist_ok=True)   # Create test data dir if missing

    # Load the training and testing CIFAR-10 datasets (download if not already present)
    dataset_train = datasets.CIFAR10(
        root=config['paths']['train_dir'],
        train=True,
        download=True,
        transform=transform
    )
    dataset_test = datasets.CIFAR10(
        root=config['paths']['test_dir'],
        train=False,
        download=True,
        transform=transform
    )

    # Create DataLoader for training and testing with support for batching, shuffling, and multiprocessing
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=config['hyperparameters']['num_workers']
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False,
        num_workers=config['hyperparameters']['num_workers']
    )
    return dataloader_train, dataloader_test


def train(model, trainloader, criterion, optimizer, device):
    model.train() # Set the model to training mode
    epochs = config['hyperparameters']['epochs']
    for epoch in range(epochs):
        running_loss = 0.0
        # Use tqdm for progress bar and live loss tracking
        with tqdm(
            trainloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            unit="batch"
        ) as progress_bar:
            for i, (inputs, labels) in enumerate(progress_bar):
                # Move data to the selected device (CPU or GPU)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # Clear gradients from previous step
                outputs = model(inputs)  # Forward pass: compute predictions
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass: compute gradients
                optimizer.step()  # Update model parameters
                running_loss += loss.item()  # Add current loss to total

                # Update progress bar with average loss so far
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': avg_loss})
        # Print average loss for this epoch
        print(f"Epoch {epoch+1} finished. Avg loss: {running_loss / len(trainloader):.3f}")
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
    dataloader_train, dataloader_test = get_loaders()

    # Create the neural network and move it to the selected device
    model = SimpleNet().to(device)

    # Define the loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (SGD with learning rate and momentum from config)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['hyperparameters']['lr'],
        momentum=config['hyperparameters']['momentum']
    )

    train(model, dataloader_train, criterion, optimizer, device)
    test(model, dataloader_test, device)

    # Save the trained model's parameters to a file
    os.makedirs(os.path.dirname(config['paths']['model_path']), exist_ok=True)  # Create weights dir if missing
    torch.save(model.state_dict(), config['paths']['model_path']) # Save the model's parameters to a pth file
    print(f"Model saved to {config['paths']['model_path']}")

if __name__ == '__main__':
    main()
