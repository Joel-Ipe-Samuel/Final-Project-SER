import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split  
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from Modules import EnhancedSERModel
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Enable cuDNN autotuner for faster convolutions
torch.backends.cudnn.benchmark = True

# Custom Dataset class
class SERDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Function to delete model files
def delete_model_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            os.remove(os.path.join(directory, filename))

# Load data and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r'Datasets\Processed Data\data.npz'
data = np.load(data_path)
X_balanced = torch.tensor(data['X'], dtype=torch.float32)
y_balanced = torch.tensor(data['y'], dtype=torch.long)

# Load label encoder and scaler
with open(r'Datasets\Processed Data\encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open(r'Datasets\Processed Data\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create dataset
dataset = SERDataset(X_balanced, y_balanced)

# Create a separate test set (20% of the data)
test_size = int(0.2 * len(dataset))
train_val_size = len(dataset) - test_size
train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

# K-Fold Cross Validation setup
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Loss with class weights
class_weights = torch.tensor([1.0 / count for count in np.bincount(y_balanced.cpu().numpy())], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Set gradient accumulation steps
accumulation_steps = 4  # Number of batches to accumulate gradients over

# Training function
def train_model(model, train_loader, optimizer):
    model.train()
    train_loss, train_correct = 0, 0
    optimizer.zero_grad()
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()

        # Update gradients after accumulating
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    # Update gradients if there are leftover batches
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_accuracy = train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader)
    return train_loss, train_accuracy

# Validation function
def validate_model(model, val_loader):
    model.eval()
    val_loss, val_correct = 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_accuracy = val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader)
    return val_loss, val_accuracy, all_labels, all_preds

# Clear old model files before starting
model_directory = r'Saved Models'
delete_model_files(model_directory)

# Cross-validation loop
all_val_losses = []
all_val_accuracies = []
best_overall_val_loss = float('inf')
best_model_fold = -1
training_losses = []
validation_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
    print(f'Fold {fold + 1}/{k}')
    
    # Create data loaders for this fold
    train_subset = Subset(train_val_dataset, train_idx)
    val_subset = Subset(train_val_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    # Initialize model, optimizer, and scheduler
    input_size = X_balanced.shape[1]
    num_classes = len(encoder.classes_)
    model = EnhancedSERModel(input_size, num_classes).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait before stopping
    trigger_times = 0  # Number of epochs with no improvement

    # Training loop
    num_epochs = 50
    fold_train_losses = []
    fold_val_losses = []
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer)
        val_loss, val_accuracy, all_labels, all_preds = validate_model(model, val_loader)

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(model_directory, f'Best Model Fold_{fold}.pth'))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("\nEarly stopping triggered")
                break

    training_losses.append(fold_train_losses)
    validation_losses.append(fold_val_losses)

    # Check if this fold's best validation loss is the best overall
    if best_val_loss < best_overall_val_loss:
        best_overall_val_loss = best_val_loss
        best_model_fold = fold
        # Save best overall model
        torch.save(model.state_dict(), os.path.join(model_directory, 'Best Overall Model.pth'))

# Load the best overall model for testing
model.load_state_dict(torch.load(os.path.join(model_directory, 'Best Overall Model.pth'), map_location=device), strict=True)
model.eval()

# Prepare test data loader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Testing function
def test_model(model, test_loader):
    model.eval()
    test_correct = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_accuracy = test_correct / len(test_loader.dataset)
    return test_accuracy, all_labels, all_preds

# Run testing and print results
test_accuracy, test_labels, test_preds = test_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}\n")
print("Classification Report on Test Set:\n")
print(classification_report(test_labels, test_preds, target_names=encoder.classes_))

# Confusion matrix for test set
conf_matrix = confusion_matrix(test_labels, test_preds)
print("Confusion Matrix on Test Set:\n")
print(conf_matrix)

# Plot training and validation loss
for fold in range(k):
    plt.plot(training_losses[fold], label=f'Train Fold {fold + 1}')
    plt.plot(validation_losses[fold], label=f'Val Fold {fold + 1}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss across Folds')
plt.legend()
plt.show()

# Plot confusion matrix
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
