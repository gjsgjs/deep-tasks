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
from qqdm import qqdm




def train():
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=False, flatten=False)
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.reshape((70000,1, 28, 28))
    x_train = torch.tensor(x_train, dtype=torch.float32)
    # 归一化到 [-1, 1] 范围
    x_train = (x_train / 255.0) * 2 - 1

    train_loader = DataLoader(x_train, batch_size=64, shuffle=True)

    # 可视化部分数据
    if args.show_images:
        examples = enumerate(train_loader)
        batch_idx, example_data = next(examples)
        # Convert tensor to numpy array and normalize to [0, 255]
        example_data = example_data.numpy()
        example_data = ((example_data + 1) / 2 * 255).astype(np.uint8)

        for i in range(6):
            img = example_data[i][0]
            cv2.imshow(f'Image {i+1}', img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

   
    # 必要目录
    log_dir = os.path.join(".", 'logs')
    ckpt_dir = os.path.join(".", 'gan_checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)



    # 初始化网络
    generator = Generator(100).to(device)
    discriminator = Discriminator(1).to(device)
    # 定义损失函数和优化器
    #
    lr = 1e-4
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr,betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr,betas=(0.5, 0.999))

    # 训练 GAN
    num_epochs = 50
    for e,epoch in enumerate(range(num_epochs)):
        progress_bar = qqdm(train_loader)
        for i, images in enumerate(progress_bar):
            images = images.to(device)
            # 训练判别器
            real_labels = torch.ones(images.size(0)).to(device)
            fake_labels = torch.zeros(images.size(0)).to(device)

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


            progress_bar.set_infos({
                'Loss_D': round(d_loss.item(), 4),
                'Loss_G': round(g_loss.item(), 4),
                'Epoch': e+1,
                })

        print(f'Epoch [{e+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
        # 保存模型参数
        generator.eval()
        z = torch.randn(10, 100).to(device)
        fake_images = generator(z)
        fake_images = fake_images.data
        fake_images = fake_images.cpu().numpy()
        fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)  # 反归一化
        # for i in range(10):
        #     img = fake_images[i][0]
        #     img_path = os.path.join(log_dir, f'epoch_{epoch}_image_{i+1}.png')
        #     cv2.imwrite(img_path, img)
        #     print(f'Saved {img_path}')
        # 创建一个空白的大图像用于拼接
        combined_image = np.zeros((28 * 2, 28 * 5), dtype=np.uint8)
        for i in range(10):
            img = fake_images[i][0]
            row = i // 5
            col = i % 5
            combined_image[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = img

        img_path = os.path.join(log_dir, f'epoch_{epoch}.png')
        cv2.imwrite(img_path, combined_image)
        print(f'Saved {img_path}')


        generator.train()
        if (e+1) % 5 == 0:
            save_model(generator, discriminator, epoch+1,d_loss.item(),g_loss.item())





def infer():
    # 推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(100).to(device)
    check_point = "gan_checkpoints\generator_epoch_35d_loss0.8335685729980469_g_loss1.5297629833221436.pth"
    generator.load_state_dict(torch.load(check_point, map_location=device))
    print(f'Loaded {check_point}')
    generator.eval()
    # print(generator)
    zz = torch.randn(100, 100).to(device)
    fake_images = generator(zz)
    fake_images = fake_images.data
    fake_images = fake_images.cpu().numpy()
    fake_images = ((fake_images + 1) / 2 * 255).astype(np.uint8)  # 反归一化
    # 创建一个空白的大图像用于拼接  
    combined_image = np.zeros((28 * 10, 28 * 10), dtype=np.uint8)
    for i in range(100):
        img = fake_images[i][0]
        row = i // 10
        col = i % 10
        combined_image[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = img

    log_dir = os.path.join(".", 'logs')
    os.makedirs(log_dir, exist_ok=True)
    img_path = os.path.join(log_dir, f'test0.png')
    cv2.imwrite(img_path, combined_image)
    print(f'Saved {img_path}')

    if args.show_fake:
        cv2.imshow(f'Generated Images ', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or infer gan model.')
    #parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help='Mode to run the script in: train or infer')
    parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer'], help='Mode to run the script in: train or infer')
    parser.add_argument('--show_images', action='store_true', help='Whether to show original images')
    parser.add_argument('--show_fake', action='store_true', help='Whether to show generated images')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'infer':
        infer()