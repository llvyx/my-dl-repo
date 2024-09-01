import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch


# 数据处理
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, valid_gestures=None, transform=None):
        self.root_dir = root_dir # 路径
        self.transform = transform # 预处理
        self.valid_gestures = valid_gestures if valid_gestures else ['01', '02', '03', '04', '05', '06'] # 需要的标签编号
        self.image_paths = [] # 图像路径
        self.labels = [] # 标签

        # 遍历每个文件夹
        for subject_folder in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject_folder)
            if os.path.isdir(subject_path):
                # 遍历手势类别文件夹
                for gesture_folder in os.listdir(subject_path):
                    gesture_code = gesture_folder[:2]  # 提取手势编号
                    if gesture_code in self.valid_gestures:
                        gesture_path = os.path.join(subject_path, gesture_folder)
                        if os.path.isdir(gesture_path):
                            # 获取所有图像路径
                            for image_file in glob.glob(os.path.join(gesture_path, '*.png')):
                                self.image_paths.append(image_file)
                                self.labels.append(int(gesture_code) - 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 设置数据集路径
root_dir = '/home/panda/dataset'

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 只加载前6种手势
valid_gestures = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# 加载数据集
dataset = HandGestureDataset(root_dir=root_dir, valid_gestures=valid_gestures, transform=transform)

# 划分数据集
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 构建模型
class CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ShallowCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.fc1(x)
        return x

class SingleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SingleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 64 * 64, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 3 * 128 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OneLayerMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(3 * 128 * 128, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 3 * 128 * 128)
        x = self.fc(x)
        return x

# 训练模型
# 超参数
num_epochs = 5
learning_rate = 0.001

# 定义模型、损失函数和优化器
model = CNN(num_classes=10) # 模型
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 将数据传输到设备
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        model.to('cuda')

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')


# 保存参数
torch.save(model.state_dict(), 'gesture_recognition_model.pth')


# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'test Accuracy: {100 * correct / total:.2f}%')