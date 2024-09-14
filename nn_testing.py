import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


TEST_DIR = 'processing_images/test'
BATCH_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data transforms (preprocessing and augmentation)
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
}

test_dataset = datasets.ImageFolder(TEST_DIR, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model architecture (ResNet-18 in this case)
def load_model(num_classes):
    model = models.resnet18(pretrained=False)  # We don't need pre-trained weights here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjusting the output layer to match our number of classes
    return model

# Test the model on the new test dataset
def test_model(model, test_loader):
    model.eval()
    test_preds = []
    test_labels = []
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)    
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate classification metrics
    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=test_loader.dataset.classes))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Load the best model's weights and architecture
MODEL_PATH = 'best_model.pth'  # Path to the saved model
NUM_CLASSES = 3  # Number of output classes (esophagus, stomach, duodenum)

# Load the model architecture
model = load_model(NUM_CLASSES)

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH))

# Assuming you have a test_loader ready
test_model(model, test_loader)
