import os
import torch
import torch.nn as nn
import cv2

# # 定义生成器和判别器
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(100, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 28*28),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.main(x).view(-1, 1, 28, 28)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(28*28, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.main(x.view(-1, 28*28))
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 1, 28, 28)
    """
    def __init__(self, in_dim, dim=32):
        super(Generator, self).__init__()
        
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 7 * 7, bias=False),  # Adjusted for 7*7
            nn.BatchNorm1d(dim * 7 * 7),
            nn.ReLU()
        )
        
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim, dim * 2),  # 7*7 to 14*14
            dconv_bn_relu(dim * 2, dim * 4),  # 14*14 to 28*28
            nn.ConvTranspose2d(dim * 4, 1, 5, 1, padding=2),  # Adjusted to output 28*28
            nn.Tanh()
        )
        
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 7, 7)  # Reshape to (N, dim*7*7)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 1, 28, 28)
    Output shape: (N, )
    """
    def __init__(self, in_dim=1, dim=32):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),  # Stride 2 for downsampling
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),  # From (28, 28) to (14, 14)
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),  # From (14, 14) to (7, 7)
            nn.Conv2d(dim * 2, dim * 4, 7),  # From (7, 7) to (1, 1)
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 4, 1, 1),  # Final layer to get output
            nn.Sigmoid()
        )
        
        self.apply(weights_init)
        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)  # Flatten the output
        return y
    

def save_model(generator, discriminator, epoch, d_loss,g_loss,path='gan_checkpoints'):
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
    torch.save(generator.state_dict(), f'{path}/generator_epoch_{epoch}d_loss{d_loss}_g_loss{g_loss}.pth')
    torch.save(discriminator.state_dict(), f'{path}/discriminator_epoch_{epoch}d_loss{d_loss}_g_loss{g_loss}.pth')
    print(f'Models saved at epoch {epoch}')


# img = torch.ones(10, 1, 28, 28)
# model = Discriminator(1)
# x = model(img)
# print(x)

# z1 = torch.randn(1, 100)
# G = Generator(100)
# G.eval()
# x = G(z1)
# print(x.shape)