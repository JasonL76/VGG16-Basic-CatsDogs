import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import albumentations as A

data_path = ".../vgg2/input/cat-and-dog"

class CatvsDog(Dataset):
    def __init__(self, data_dir, transforms=None):
        catpaths = os.path.join(data_dir, 'cats')
        dogpaths = os.path.join(data_dir, 'dogs')
        cats = os.listdir(catpaths)
        dogs = os.listdir(dogpaths)
        self.images = [(os.path.join(catpaths, cats[i]), 0) for i in range(len(cats)) if cats[i].endswith('.jpg')]
        self.images.extend([(os.path.join(dogpaths, dogs[i]), 1) for i in range(len(dogs)) if dogs[i].endswith('.jpg')])
        self.transforms = transforms
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index][0]).convert('RGB'))
        y = self.images[index][1]
        if self.transforms is not None:
            augmentations = self.transforms(image=img)
            img = augmentations["image"]
        return img, y

train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def get_dogandcat(image_dir, train_transforms=None, test_transforms=None, batch_size=1, shuffle=True, pin_memory=True):
    data = CatvsDog(image_dir, transforms=None)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    train_dataset.dataset.transforms = train_transforms
    test_dataset.dataset.transforms = test_transforms

    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return train_batch, test_batch

# Hyperparameters
learning_rate = 1e-4
num_epochs = 10
batch_size = 16

train_batch, test_batch = get_dogandcat(data_path, train_transforms=train_transforms, test_transforms=test_transforms, batch_size=batch_size, shuffle=True, pin_memory=True)

for i, j in train_batch:
    img = np.transpose(i[0].numpy(), (1, 2, 0))
    print(j)
    plt.imshow(img)
    plt.show()
    break

class N_conv(nn.Module):
    def __init__(self, in_channels, out_channels, N=2):
        super(N_conv, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
        model.append(nn.ReLU(True))
        for i in range(N - 1):
            model.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
            model.append(nn.ReLU(True))
        model.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)

class Vgg16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_weights=True):
        super(Vgg16, self).__init__()
        self.conv1 = N_conv(3, 64)
        self.conv2 = N_conv(64, 128)
        self.conv3 = N_conv(128, 256, N=3)
        self.conv4 = N_conv(256, 512, N=3)
        self.conv5 = N_conv(512, 512, N=3)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.linear1 = nn.Linear(512 * 7 * 7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(4096, 2)
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = Vgg16(3, 2).to(device)
from torchsummary import summary
summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=5, verbose=True)

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(enumerate(train_batch), total=len(train_batch))
    running_loss = 0
    for batch_idx, (data, targets) in loop:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        loop.set_postfix(loss=loss.item())
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_batch)}")
    
    val_loss = running_loss / len(train_batch)
    early_stopping(val_loss)

    if early_stopping.early_stop:
        print("Early stopping")
        break

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    accuracy = num_correct / num_samples
    model.train()
    return accuracy

train_acc = check_accuracy(train_batch, model)
test_acc = check_accuracy(test_batch, model)
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
