import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

transform = transforms.Compose([
    transforms.Grayscale(),  # make sure images are in gray scale 
    transforms.Resize((28, 28)),  # make sure images are of size 28*28
    transforms.ToTensor(),  # convert images to PyTorch tensor
])

# Lenet-5 for 28*28
# 2 conv layers + 2 pooling layers 
# 3 fully connected layers 
# activation function: Relu 
# Softmax was removed to enhance training
class LeNet5_28(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # conv1: from 28*28*1 to 24*24*6
        self.pool1 = nn.AvgPool2d(2, 2)                 # pool1: from 24*24*6 to 12*12*6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # conv2: from 12*12*6 to 8*8*16
        self.pool2 = nn.AvgPool2d(2, 2)                 # pool2: from 8*8*16 to 4*4*16 
        self.fc1   = nn.Linear(16*4*4, 120)             # fc1: from 256 to 120
        self.fc2   = nn.Linear(120, 84)                 # fc2: from 120 to 84
        self.fc3   = nn.Linear(84, 10)                  # fc3: from 84 to 10 (10 classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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


# Variation of Lenet-5 for 28*28
# number of filters increased for conv layers
# conv1 24 filters instead of 6
# conv2 64 filters instead of 16
class LeNet5_28_more_filters(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5)     # conv1: from 28*28*1 to 24*24*32
        self.pool1 = nn.AvgPool2d(2, 2)                 # pool1: from 24*24*24 to 12*12*24
        self.conv2 = nn.Conv2d(24, 64, kernel_size=5)    # conv2: from 12*12*24 to 8*8*64
        self.pool2 = nn.AvgPool2d(2, 2)                 # pool2: from 8*8*64 to 4*4*64
        self.fc1   = nn.Linear(64*4*4, 120)             # fc1: from 1024 to 120
        self.fc2   = nn.Linear(120, 84)                 # fc2: from 120 to 84
        self.fc3   = nn.Linear(84, 10)                  # fc3: from 84 to 10 (10 classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64*4*4)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x) # produces 10 logits 


# Variation of Lenet-5 for 28*28
# use max pooling instead of average pooling 
class LeNet5_28_maxpooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # conv1: from 28*28*1 to 24*24*6
        self.pool1 = nn.MaxPool2d(2, 2)                 # pool1: from 24*24*6 to 12*12*6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # conv2: from 12*12*6 to 8*8*16
        self.pool2 = nn.MaxPool2d(2, 2)                 # pool2: from 8*8*16 to 4*4*16 
        self.fc1   = nn.Linear(16*4*4, 120)             # fc1: from 256 to 120
        self.fc2   = nn.Linear(120, 84)                 # fc2: from 120 to 84
        self.fc3   = nn.Linear(84, 10)                  # fc3: from 84 to 10 (10 classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x) # produces 10 logits 
    

# Variation of Lenet-5 for 28*28
# add batch normalization layer after each conv layer
class LeNet5_28_batch_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # conv1: from 28*28*1 to 24*24*6
        self.bn1   = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(2, 2)                 # pool1: from 24*24*6 to 12*12*6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # conv2: from 12*12*6 to 8*8*16
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(2, 2)                 # pool2: from 8*8*16 to 4*4*16 
        self.fc1   = nn.Linear(16*4*4, 120)             # fc1: from 256 to 120
        self.fc2   = nn.Linear(120, 84)                 # fc2: from 120 to 84
        self.fc3   = nn.Linear(84, 10)                  # fc3: from 84 to 10 (10 classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16*4*4)  # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x) # produces 10 logits 


# Load dataset 1000 images for training per digit and 200 images for testing per digit
train_set = ImageFolder(root='reduced_MNIST/train_data', transform=transform)
test_set  = ImageFolder(root='reduced_MNIST/test_data',  transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

# Verify train and test sets sizes
print(f"Train set size: {len(train_set)}")  # should be 10000
print(f"Test set size:  {len(test_set)}")   # should be 2000

# Variation 1
print("Model 1: Lenet-5 for 28*28 (base model)")
model_1 = LeNet5_28() # base model
print("Model 1 training:")
train_accuracy_1, train_time_1 = train_model(model_1, train_loader)
print("Model 1 test:")
test_accuracy_1, test_time_1 = evaluate_model(model_1, test_loader)

# Variation 2
print("Model 2: Lenet-5 for 28*28 with more filters")
model_2 = LeNet5_28_more_filters()
print("Model 2 training:")
train_accuracy_2, train_time_2 = train_model(model_2, train_loader)
print("Model 2 test:")
test_accuracy_2, test_time_2 = evaluate_model(model_2, test_loader)

# Variation 3
print("Model 3: Lenet-5 for 28*28 with max pooling")
model_3 = LeNet5_28_maxpooling()
print("Model 3 training:")
train_accuracy_3, train_time_3 = train_model(model_3, train_loader)
print("Model 3 test:")
test_accuracy_3, test_time_3 = evaluate_model(model_3, test_loader)

# Variation 4
print("Model 4: Lenet-5 for 28*28 with batch norm")
model_4 = LeNet5_28_batch_norm()
print("Model 4 training:")
train_accuracy_4, train_time_4 = train_model(model_4, train_loader)
print("Model 4 test:")
test_accuracy_4, test_time_4 = evaluate_model(model_4, test_loader)


print("\n========== RESULTS TABLE ==========")
print(f"{'Variation':<18}{'Train Acc':>12}{'Train Time':>15}{'Test Acc':>12}{'Test Time':>15}")
print("-" * 72)
print(f"{'V1 (Base)':<18}{train_accuracy_1:>11.1f}%{train_time_1:>14.1f} ms{test_accuracy_1:>11.1f}%{test_time_1:>14.1f} ms")
print(f"{'V2 (More filters)':<18}{train_accuracy_2:>11.1f}%{train_time_2:>14.1f} ms{test_accuracy_2:>11.1f}%{test_time_2:>14.1f} ms")
print(f"{'V3 (Max pooling)':<18}{train_accuracy_3:>11.1f}%{train_time_3:>14.1f} ms{test_accuracy_3:>11.1f}%{test_time_3:>14.1f} ms")
print(f"{'V4 (Batch norm)':<18}{train_accuracy_4:>11.1f}%{train_time_4:>14.1f} ms{test_accuracy_4:>11.1f}%{test_time_4:>14.1f} ms")