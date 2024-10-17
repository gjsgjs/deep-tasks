import argparse
from torch import optim
import torch
from torch.utils.data import TensorDataset, DataLoader

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import pickle
import os
import numpy as np

from get_minist import load_mnist
import cv2
from minist import Generator, Discriminator,save_model
import torch.nn as nn





# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=False, flatten=False)
x_train = np.concatenate((x_train, x_test), axis=0)
x_train = x_train.reshape((70000, 28, 28))
x_train = torch.tensor(x_train, dtype=torch.float32)
# 归一化到 [-1, 1] 范围
x_train = (x_train / 255.0) * 2 - 1

train_loader = DataLoader(x_train, batch_size=64, shuffle=True)

# 可视化部分数据
if False:
    examples = enumerate(train_loader)
    batch_idx, example_data = next(examples)
    # Convert tensor to numpy array and normalize to [0, 255]
    example_data = example_data.numpy()
    example_data = ((example_data + 1) / 2 * 255).astype(np.uint8)

    for i in range(6):
        img = example_data[i]
        cv2.imshow(f'Image {i+1}', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# 初始化网络
generator = Generator().to(device)
discriminator = Discriminator().to(device)
# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 训练 GAN
num_epochs = 50
for epoch in range(num_epochs):
    for i, images in enumerate(train_loader):
        images = images.to(device)
        # 训练判别器
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(z)
        # 只更新判别器，而不更新生成器
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        z = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    # 保存模型参数
    save_model(generator, discriminator, epoch+1)

# 生成新图片
z = torch.randn(64, 100).to(device)
fake_images = generator(z)
fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
fake_images = fake_images.data

fake_images = fake_images.cpu().numpy()
fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)  # 反归一化

for i in range(6):
    img = fake_images[i][0]
    cv2.imshow(f'Generated Image {i+1}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# 线性插值
z1 = torch.randn(1, 100).to(device)
z2 = torch.randn(1, 100).to(device)
interpolated_images = []
for alpha in np.linspace(0, 1, 10):
    z = alpha * z1 + (1 - alpha) * z2
    interpolated_image = generator(z).view(1, 28, 28).data.cpu().numpy()
    interpolated_image = ((interpolated_image + 1) / 2 * 255).astype(np.uint8)  # 反归一化
    interpolated_images.append(interpolated_image)

# 可视化插值图片
for i in range(10):
    img = interpolated_images[i][0]
    cv2.imshow(f'Interpolated Image {i+1}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

if False:
    # 推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("gan_checkpoints\generator_epoch_5.pth", map_location=device))
    generator.eval()


    z = torch.randn(10, 100).to(device)
    fake_images = generator(z)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    fake_images = fake_images.data.cpu().numpy()
    fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)  # 反归一化

    for i in range(10):
        img = fake_images[i][0]
        cv2.imshow(f'Generated Image {i+1}', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

