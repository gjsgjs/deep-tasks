import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from minist import ViT, get_cosine_schedule_with_warmup
from qqdm import qqdm


# 数据目录
train_data_dir = 'CIFAR10_imbalanced'
test_data_dir = 'CIFAR10_balance'


#
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#
transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#
batch_size = 128
warmup_steps = 1000
total_steps = 22000
# 加载训练数据集
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试数据集
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# 查看训练集的图片和标签
if False:
    def show_images_cv2(images, labels, classes):
        images = ((images + 1) / 2 * 255)
        for i in range(len(images)):
            img = images[i].numpy().transpose((1, 2, 0)).astype('uint8')
            label = classes[labels[i]]
            print(label)
            cv2.imshow(label, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # 获取一个批次的训练数据
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # 显示图片和标签
    show_images_cv2(images, labels, train_dataset.classes)


# 训练模型并评估性能
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT().to(device)
# 计算类权重,加权损失函数 写一个args
class_counts = torch.tensor([train_dataset.targets.count(i) for i in range(10)])
class_weights = 1. / class_counts
class_weights = class_weights.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(class_weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# 训练模型
num_epochs = 80
step = 0
for e,epoch in enumerate(range(num_epochs)):
    progress_bar = qqdm(train_loader)
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # 更新学习率
        train_loss += loss.item()
        progress_bar.set_infos({
            'train_loss': loss.item(),
            'Epoch': e+1,
            'Step': step,
        })
        step += 1
    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}')

    # 评估模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch+1}, Test Loss: {test_loss/len(test_loader)}, Accuracy: {100 * correct / total}%')

