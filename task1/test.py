import torch
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

img_path = '/mnt/d/Panda/Pictures/Camera Roll/WIN_20240901_14_12_13_Pro.jpg'  # 图像路径

class CNN(nn.Module):
    def __init__(self, num_classes=10):
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
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = CNN(num_classes=10)
model.load_state_dict(torch.load('gesture_recognition_model.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 定义手势类别
gesture_classes = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

# 预处理图像
img = Image.open(img_path)
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output.data, 1)
predicted_class = gesture_classes[predicted.item()]
print(f'Predicted Gesture: {predicted_class}')

# 读取图像用于显示（OpenCV使用BGR格式）
image = cv2.imread(img_path)

# 在图像上标注预测结果
cv2.putText(image, f'Predicted Gesture: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
