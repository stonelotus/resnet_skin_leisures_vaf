import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Set the paths to your dataset and CSV file
dataset_path = '/home/stonex/private/facultate/vaf/ceva/nebun'
csv_path = '/home/stonex/private/facultate/vaf/ceva/HAM10000_metadata.csv'

# Read the CSV file
df = pd.read_csv(csv_path)

# Preprocess the labels
le = preprocessing.LabelEncoder()
df['dx'] = le.fit_transform(df['dx'])

# Define the custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 1]) + '.jpg'
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.df.iloc[idx, 2])
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split the dataframe into train and test dataframes
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_data = CustomImageDataset(train_df, dataset_path, transformations)
test_data = CustomImageDataset(test_df, dataset_path, transformations)

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader =  DataLoader(test_data,  batch_size=32, shuffle=False)

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the model weights
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier layer at the end of the model
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 7)  # Replace '7' with the number of classes in your dataset

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (only the parameters of the classifier layer will be updated)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
num_epochs = 25

best_acc = 0

# Training the model
print("Will epoch ....")
for epoch in range(num_epochs):
    print("Starting epoch: " + str(epoch))
    for i, (images, labels) in enumerate(train_loader):
        # images = images.to(device)
        # labels = labels.to(device)
        
        # Forward pass
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
    acc = validate(model, test_loader)
    if acc > best_acc:
      best_acc = acc
      torch.save(model.state_dict(), 'model.ckpt')
# Save your model if you wish
torch.save(model.state_dict(), 'model.ckpt')

# Test your model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy of the model on the test images: {100 * correct / total}%")



def validate(model, dataloader):
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Set model back to train mode
    return 100 * correct / total