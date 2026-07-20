import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

transform = transforms.Compose([
    transforms.Resize((227, 227)), # resize to 227*227 for AlexNet
    transforms.ToTensor(), # convert images to PyTorch tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # normalize values to [-1,1]
])

# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)  # conv1: from 227*227*3 to 55*55*96
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)                  # pool1: from 55*55*96 to 27*27*96
        self.bn1   = nn.BatchNorm2d(96) 

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2) # conv2: from 27*27*96 to 27*27*256
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)                  # pool2: from 27*27*256 to 13*13*256
        self.bn2   = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)# conv3: from 13*13*256 to 13*13*384
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)# conv4: from 13*13*384 to 13*13*384
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)# conv5: from 13*13*384 to 13*13*256
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)                  # pool3: from 13*13*256 to 6*6*256

        self.fc1   = nn.Linear(6*6*256, 1024)                               # fc1: from 9216 to 1024
        self.fc2   = nn.Linear(1024, 512)                                   # fc2: from 1024 to 512
        self.fc3   = nn.Linear(512, 10)                                     # fc3: from 512 to 10 (10 classes)

        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool3(self.relu(self.conv5(x)))

        x = x.view(-1, 256*6*6) # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x) # produces 10 logits 


# Attention module: spatial attention
# class attention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)   # conv: from H*W*2 to H*W*1
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)    # output size: H*W*1
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # output size: H*W*1
#         attn = torch.cat([avg_out, max_out], dim=1)     # output size: H*W*2
#         attn = self.sigmoid(self.conv(attn))            # output size: H*W*1
#         return x * attn # output size: H*W*C (input size)


# Attention module: separable attention (frequency and time)
class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_conv = nn.Conv2d(2, 1, kernel_size=(7,1), padding=(3,0))  # vertical only
        self.time_conv = nn.Conv2d(2, 1, kernel_size=(1,7), padding=(0,3))  # horizontal only
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # output size: H*W*1
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # output size: H*W*1
        attn = torch.cat([avg_out, max_out], dim=1)     # output size: H*W*2
        attn_freq = self.sigmoid(self.freq_conv(attn))
        attn_time = self.sigmoid(self.time_conv(attn))
        x = x * attn_freq  # attend to frequency bands
        x = x * attn_time  # attend to time regions
        return x # output size: H*W*C (input size)


# AlexNet with attention
class AlexNet_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)  # conv1: from 227*227*3 to 55*55*96
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)                  # pool1: from 55*55*96 to 27*27*96
        self.bn1   = nn.BatchNorm2d(96) 

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2) # conv2: from 27*27*96 to 27*27*256
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)                  # pool2: from 27*27*256 to 13*13*256
        self.bn2   = nn.BatchNorm2d(256)
        self.attention = attention()                                        # attention after pool2
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)# conv3: from 13*13*256 to 13*13*384
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)# conv4: from 13*13*384 to 13*13*384
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)# conv5: from 13*13*384 to 13*13*256
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)                  # pool3: from 13*13*256 to 6*6*256

        self.fc1   = nn.Linear(6*6*256, 1024)                               # fc1: from 9216 to 1024
        self.fc2   = nn.Linear(1024, 512)                                   # fc2: from 1024 to 512
        self.fc3   = nn.Linear(512, 10)                                     # fc3: from 512 to 10 (10 classes)

        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.attention(x)   # apply attention
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool3(self.relu(self.conv5(x)))

        x = x.view(-1, 256*6*6) # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x) # produces 10 logits 

    
# Train model
def train_model(model, loader, epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_accuracy = round(100 * correct / total, 1)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {train_accuracy}%")

    train_time = (time.time() - start) * 1000
    print(f"Total Training Time: {train_time:.1f} ms")
    return train_accuracy, round(train_time, 1)

# Test model
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            _, predicted = torch.max(model(images), 1)  # predicted class index
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    test_time = (time.time() - start) * 1000
    accuracy = round(100 * correct / total, 1)
    print(f"Test Accuracy: {accuracy}% | Test Time: {test_time:.1f} ms")
    return accuracy, round(test_time, 1)


# Load dataset (spectrogram images from assignment 2)
train_set = ImageFolder(root='spectrograms-dataset/Train', transform=transform) 
test_set  = ImageFolder(root='spectrograms-dataset/Test',  transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

# Verify train and test sets sizes
print(f"Train set size: {len(train_set)}")  # should be 1200
print(f"Test set size:  {len(test_set)}")   # should be 300

print("Model 1: AlexNet")
model_1 = AlexNet() 
print("Model 1 training:")
train_accuracy_1, train_time_1 = train_model(model_1, train_loader)
print("Model 1 test:")
test_accuracy_1, test_time_1 = evaluate_model(model_1, test_loader)

print("Model 2: AlexNet with attention")
model_2 = AlexNet_attention()
print("Model 2 training:")
train_accuracy_2, train_time_2 = train_model(model_2, train_loader)
print("Model 2 test:")
test_accuracy_2, test_time_2 = evaluate_model(model_2, test_loader)

print("\n========== RESULTS TABLE ==========")
print(f"{'Model':<26} {'Train Acc':>10} {'Train Time':>14} {'Test Acc':>10} {'Test Time':>14}")
print("-" * 80)
print(f"{'AlexNet':<26} {train_accuracy_1:>9.1f}% {train_time_1:>11.1f} ms {test_accuracy_1:>9.1f}% {test_time_1:>11.1f} ms")
print(f"{'AlexNet with attention':<26} {train_accuracy_2:>9.1f}% {train_time_2:>11.1f} ms {test_accuracy_2:>9.1f}% {test_time_2:>11.1f} ms")
