from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Directories
TRAIN_DIR = 'processing_images/train'
VAL_DIR = 'processing_images/val'

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Number of target classes
MODEL_PATH = 'best_model.pth'  # Path to save the best model
# Constants for normalization (mean and std of ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data transforms (preprocessing and augmentation)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
}

# Datasets and loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model - using pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Modify the final layer to fit our number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Variables to track the best model
best_model_wts = None
best_f1 = 0.0

# Store losses and accuracy for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training and validation loop
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct.double() / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())
    print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            val_total += labels.size(0)
            
            # Store predictions and true labels for F1-score calculation
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_acc = val_correct.double() / val_total
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc.item())
    
    print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}, F1-Score: {val_f1:.4f}')
    
    # Check if this is the best model so far, and save it
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_wts = model.state_dict()
        torch.save(model.state_dict(), MODEL_PATH)  # Save the best model
        print(f"Best model saved with F1-Score: {best_f1:.4f}")

# Load the best model weights after training
if best_model_wts:
    model.load_state_dict(best_model_wts)

# Plotting loss and accuracy curves
epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 4))
# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()
