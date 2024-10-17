import os
import torch
import torch.nn as nn
import cv2

# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x.view(-1, 28*28))
    



    

def save_model(generator, discriminator, epoch, path='gan_checkpoints'):
    """
    保存生成器和判别器的模型参数

    Parameters
    ----------
    generator : nn.Module
        生成器模型
    discriminator : nn.Module
        判别器模型
    epoch : int
        当前训练的轮数
    path : str
        保存模型参数的路径
    """
    torch.save(generator.state_dict(), f'{path}/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'{path}/discriminator_epoch_{epoch}.pth')
    print(f'Models saved at epoch {epoch}')

