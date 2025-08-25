import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define image transformations: resizing, converting to tensor, and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB channels
])

# Load training and validation datasets using ImageFolder
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

# Define DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a sample CNN model
class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()
        # Convolutional layers with ReLU activations and MaxPooling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Conv layer: 3 input channels, 16 filters
            nn.ReLU(),                                   # ReLU activation
            nn.MaxPool2d(2, 2),                          # MaxPooling with 2x2 kernel
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Conv layer: 16 -> 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Conv layer: 32 -> 64 filters
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),               # Flatten the 3D feature maps to 1D vector
            nn.Linear(64*28*28, 128),  # Fully connected layer with 128 units
            nn.ReLU(),
            nn.Linear(128, 2)           # Output layer for 2 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass input through convolutional layers
        x = self.fc_layers(x)    # Pass through fully connected layers
        return x

# Initialize model, loss function, and optimizer
model = SampleCNN()
criterion = nn.CrossEntropyLoss()     # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

num_epochs = 10

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()           # Reset gradients
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation on validation set
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")
