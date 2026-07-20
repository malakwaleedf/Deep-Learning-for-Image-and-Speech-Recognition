import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((227, 227)), # resize to 227*227 for AlexNet
    transforms.ToTensor(), # convert images to PyTorch tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # normalize values to [-1,1]
])

# Converts audio to spectrogram
def audio_to_spectrogram(wav_path, out_path):
    audio, sampling_rate = librosa.load(wav_path, sr=None)
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(2.27, 2.27), dpi=100)
    librosa.display.specshow(spectrogram, sr=sampling_rate, ax=ax)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Extract digit label from .wav file name 
def get_digit(filename):
    stem = Path(filename).stem 
    return stem[-1] # returns last character (the digit)

# Convert audio dataset to spectrogram
def convert_dataset(audio_root, spectrogram_root):
    audio_root = Path(audio_root)
    spectrogram_root = Path(spectrogram_root)

    for split in ['Train', 'Test']:
        in_dir = audio_root / split
        wav_files = list(in_dir.glob('*.wav')) # get all .wav files 
        print(f"[{split}] found {len(wav_files)} files") # 1200 for train, 300 for test

        for wav_path in wav_files:
            digit = get_digit(wav_path.name)
            # create output folder per digit
            out_dir = spectrogram_root / split / digit
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (wav_path.stem + '.png')

            if out_path.exists(): # avoid duplicates 
                continue

            try:
                audio_to_spectrogram(str(wav_path), str(out_path))
            except Exception as e:
                print(f"  ERROR on {wav_path.name}: {e}")

    print("Original dataset is conveted to spectrogram\n")

# Apply speech augmentation
# speed up, speed down, add noise 
def augment_speech(audio, aug_type):

    if aug_type == 'speed_up':
        return librosa.effects.time_stretch(audio, rate=1.03)   # rate > 1 -> faster playback
 
    elif aug_type == 'speed_down':
        return librosa.effects.time_stretch(audio, rate=0.97)   # rate < 1 -> slower playback
 
    elif aug_type == 'noise':
        rms   = np.sqrt(np.mean(audio ** 2))
        noise = np.random.normal(0, rms * 0.1, size=audio.shape)
        return audio + noise
 
    else:
        raise ValueError(f"  Unknown speech augmentation: {aug_type}")
    
# Convert augmented audio to spectrogram and add to original spectrogram dataset
def add_speech_augmentations(audio_root, spectrogram_root, augmented_spectrogram_root):
    audio_root = Path(audio_root)
    spectrogram_root  = Path(spectrogram_root)
    augmented_spectrogram_root  = Path(augmented_spectrogram_root)
    aug_types  = ['speed_up', 'speed_down', 'noise']

    # copy original spectrograms
    print(f"Copying {spectrogram_root} to {augmented_spectrogram_root}")
    for png in spectrogram_root.rglob('*.png'):
        dest = augmented_spectrogram_root / png.relative_to(spectrogram_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(png, dest)
 
    # generate speech augmented spectrograms (for train set only)
    wav_files = list((audio_root / 'Train').glob('*.wav'))
    print(f"[Speech Aug] processing {len(wav_files)} training files") # 1200
 
    for wav_path in wav_files:
        digit   = get_digit(wav_path.name)
        out_dir = augmented_spectrogram_root / 'Train' / digit
        out_dir.mkdir(parents=True, exist_ok=True)
 
        audio, sampling_rate = librosa.load(str(wav_path), sr=None)
 
        for aug in aug_types:
            out_path = out_dir / f"{wav_path.stem}_{aug}.png"
            if out_path.exists():
                continue

            aug_audio = augment_speech(audio, aug)
            spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(aug_audio)), ref=np.max)
            fig, ax = plt.subplots(figsize=(2.27, 2.27), dpi=100)
            librosa.display.specshow(spectrogram, sr=sampling_rate, ax=ax)
            ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    print("Speech augmentations added to original dataset\n")

# Apply image augmentation 
# squeeze, expand, add  noise 
def augment_spectrogram_image(img_array, aug_type):
    img  = Image.fromarray(img_array)
    W, H = img.size
 
    if aug_type == 'squeeze':
        # shrink width by 3%, zero-pad back to original size
        new_w   = int(W * 0.97)
        resized = img.resize((new_w, H), Image.BILINEAR)
        canvas  = Image.new('RGB', (W, H), (0, 0, 0))
        canvas.paste(resized, ((W - new_w) // 2, 0))
        return np.array(canvas)
 
    elif aug_type == 'expand':
        # grow width by 3%, centre-crop back to original size
        new_w   = int(W * 1.03)
        resized = img.resize((new_w, H), Image.BILINEAR)
        left    = (new_w - W) // 2
        return np.array(resized)[:, left:left + W, :]
 
    elif aug_type == 'img_noise':
        arr   = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, arr.shape)
        return np.clip(arr + noise, 0, 255).astype(np.uint8)
 
    else:
        raise ValueError(f"  Unknown image augmentation: {aug_type}")

# Apply augmentation to spectrum images and add to original spectrogram dataset
def add_image_augmentations(spectrogram_root, augmented_spectrogram_root):
    spectrogram_root = Path(spectrogram_root)
    augmented_spectrogram_root = Path(augmented_spectrogram_root)
    aug_types = ['squeeze', 'expand', 'img_noise']
 
    # copy original spectrograms
    print(f"Copying {spectrogram_root} to {augmented_spectrogram_root}")
    for png in spectrogram_root.rglob('*.png'):
        dest = augmented_spectrogram_root / png.relative_to(spectrogram_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(png, dest)
 
    # generate image augmented spectrograms (for train set only)
    print(f"[Image Aug] processing {len(list((spectrogram_root / 'Train').rglob('*.png')))} training files") # 1200
    for digit_dir in sorted((spectrogram_root / 'Train').iterdir()):
        if not digit_dir.is_dir():
            continue
  
        originals = [p for p in digit_dir.glob('*.png')]
 
        for png_path in originals:
            img_arr = np.array(Image.open(png_path).convert('RGB'))

            out_dir = augmented_spectrogram_root / 'Train' / digit_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for aug in aug_types:
                out_path = out_dir / f"{png_path.stem}_{aug}.png"
                if out_path.exists():
                    continue
                aug_arr = augment_spectrogram_image(img_arr, aug)
                Image.fromarray(aug_arr).save(out_path)

    print("Image augmentations added to original dataset\n")

# Add both speech and image augmented spectrograms to original spectrogram dataset
def add_combined_augmentations(spectrogram_root, speech_augmented_spectrogram_root, image_augmented_spectrogram_root, combined_augmented_spectrogram_root):
    spectrogram_root = Path(spectrogram_root)
    speech_augmented_spectrogram_root = Path(speech_augmented_spectrogram_root)
    image_augmented_spectrogram_root = Path(image_augmented_spectrogram_root)
    combined_augmented_spectrogram_root = Path(combined_augmented_spectrogram_root)

    # copy original spectrograms
    print(f"Copying {spectrogram_root} to {combined_augmented_spectrogram_root}")
    for png in spectrogram_root.rglob('*.png'):
        dest = combined_augmented_spectrogram_root / png.relative_to(spectrogram_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(png, dest)

    # copy speech augmented spectrograms
    print(f"Copying {speech_augmented_spectrogram_root} to {combined_augmented_spectrogram_root}")
    for png in speech_augmented_spectrogram_root.rglob('*.png'):
        dest = combined_augmented_spectrogram_root / png.relative_to(speech_augmented_spectrogram_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(png, dest)

    # copy image augmented spectrograms
    print(f"Copying {image_augmented_spectrogram_root} to {combined_augmented_spectrogram_root}")
    for png in image_augmented_spectrogram_root.rglob('*.png'):
        dest = combined_augmented_spectrogram_root / png.relative_to(image_augmented_spectrogram_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(png, dest)

    print("Speech and image augmentations added to original dataset\n")


# AlexNet
# 5 conv layers + 3 pooling layers + 2 batch norm layers
# 3 fully connected layers 
# activation function: Relu 
# 2 dropouts
# Softmax was removed to enhance training
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
    
# Train model
def train_model(model, loader, epochs=10, lr=0.001):
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


convert_dataset(audio_root = 'audio-dataset', spectrogram_root = 'spectrograms-dataset')
add_speech_augmentations (audio_root = 'audio-dataset', spectrogram_root = 'spectrograms-dataset', 
                          augmented_spectrogram_root = 'speech-aug-spectrograms-dataset')
add_image_augmentations(spectrogram_root = 'spectrograms-dataset', augmented_spectrogram_root = 'image-aug-spectrograms-dataset')
add_combined_augmentations(spectrogram_root = 'spectrograms-dataset', speech_augmented_spectrogram_root = 'speech-aug-spectrograms-dataset',
                           image_augmented_spectrogram_root = 'image-aug-spectrograms-dataset', combined_augmented_spectrogram_root = 'combined-aug-spectrogram-dataset')


# Variation 1
print("Variation 1: training AlexNet model using original spectrograms dataset")

# Load dataset
train_set_1 = ImageFolder('spectrograms-dataset/Train', transform=transform)
test_set_1  = ImageFolder('spectrograms-dataset/Test',  transform=transform)

train_loader_1 = DataLoader(train_set_1, batch_size=32, shuffle=True)
test_loader_1  = DataLoader(test_set_1,  batch_size=32, shuffle=False)

# Verify train and test sets sizes
print(f"Train set size: {len(train_set_1)}")  # should be 1200
print(f"Test set size:  {len(test_set_1)}")   # should be 300

# Model
model_1 = AlexNet() 

# Train and test
print("Variation 1 training:")
train_accuracy_1, train_time_1 = train_model(model_1, train_loader_1)
print("Variation 1 test:")
test_accuracy_1, test_time_1 = evaluate_model(model_1, test_loader_1)

# Variation 2
print("Variation 2: training AlexNet model using original spectrograms dataset and speech augmented spectrograms")

# Load speech augmented dataset
train_set_2 = ImageFolder('speech-aug-spectrograms-dataset/Train', transform=transform)
test_set_2  = ImageFolder('speech-aug-spectrograms-dataset/Test',  transform=transform)

train_loader_2 = DataLoader(train_set_2, batch_size=32, shuffle=True)
test_loader_2  = DataLoader(test_set_2,  batch_size=32, shuffle=False)

# Verify train and test sets sizes
print(f"Train set size: {len(train_set_2)}")  # should be 4800
print(f"Test set size:  {len(test_set_2)}")   # should be 300

# Model
model_2 = AlexNet() 

# Train and test
print("Variation 2 training:")
train_accuracy_2, train_time_2 = train_model(model_2, train_loader_2)
print("Variation 2 test:")
test_accuracy_2, test_time_2 = evaluate_model(model_2, test_loader_2)

# Variation 3
print("Variation 3: training AlexNet model using original spectrograms dataset and image augmented spectrograms")

# Load image augmented dataset
train_set_3 = ImageFolder('image-aug-spectrograms-dataset/Train', transform=transform)
test_set_3 = ImageFolder('image-aug-spectrograms-dataset/Test',  transform=transform)

train_loader_3 = DataLoader(train_set_3, batch_size=32, shuffle=True)
test_loader_3  = DataLoader(test_set_3,  batch_size=32, shuffle=False)

# Verify train and test sets sizes
print(f"Train set size: {len(train_set_3)}")  # should be 4800
print(f"Test set size:  {len(test_set_3)}")   # should be 300

# Model
model_3 = AlexNet() 

# Train and test
print("Variation 3 training:")
train_accuracy_3, train_time_3 = train_model(model_3, train_loader_3)
print("Variation 3 test:")
test_accuracy_3, test_time_3 = evaluate_model(model_3, test_loader_3)

# Variation 4
print("Variation 4: training AlexNet model using original spectrograms dataset, speech augmented spectrograms and image augmented spectrograms")

# Load combined augmented dataset
train_set_4 = ImageFolder('combined-aug-spectrogram-dataset/Train', transform=transform)
test_set_4 = ImageFolder('combined-aug-spectrogram-dataset/Test',  transform=transform)

train_loader_4 = DataLoader(train_set_4, batch_size=32, shuffle=True)
test_loader_4  = DataLoader(test_set_4,  batch_size=32, shuffle=False)

# Verify train and test sets sizes
print(f"Train set size: {len(train_set_4)}")  # should be 8400
print(f"Test set size:  {len(test_set_4)}")   # should be 300

# Model
model_4 = AlexNet() 

# Train and test
print("Variation 4 training:")
train_accuracy_4, train_time_4 = train_model(model_4, train_loader_4, epochs=15)
print("Variation 4 test:")
test_accuracy_4, test_time_4 = evaluate_model(model_4, test_loader_4)

print("\n========== RESULTS TABLE ==========")
print(f"{'Variation':<18}{'Train Acc':>12}{'Train Time':>15}{'Test Acc':>12}{'Test Time':>15}")
print("-" * 72)
print(f"{'V1 (Original)':<18}{train_accuracy_1:>11.1f}%{train_time_1:>14.1f} ms{test_accuracy_1:>11.1f}%{test_time_1:>14.1f} ms")
print(f"{'V2 (Speech aug)':<18}{train_accuracy_2:>11.1f}%{train_time_2:>14.1f} ms{test_accuracy_2:>11.1f}%{test_time_2:>14.1f} ms")
print(f"{'V3 (Image aug)':<18}{train_accuracy_3:>11.1f}%{train_time_3:>14.1f} ms{test_accuracy_3:>11.1f}%{test_time_3:>14.1f} ms")
print(f"{'V4 (Combined aug)':<18}{train_accuracy_4:>11.1f}%{train_time_4:>14.1f} ms{test_accuracy_4:>11.1f}%{test_time_4:>14.1f} ms")
print("\n")

