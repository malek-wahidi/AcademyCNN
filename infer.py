import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

from model import BetterNet
import os

# CIFAR-10 classes
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Image path from CLI or hardcoded fallback
image_path = sys.argv[1] if len(sys.argv) > 1 else 'animal.jpg'

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    image = Image.open(image_path).convert('RGB')
except Exception as e:
    print(f"Error opening image: {e}")
    exit(1)

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetterNet().to(device)
model_path = 'weights/best_model.pth'

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    class_idx = predicted.item()
    print(f"Predicted class: {classes[class_idx]}")
