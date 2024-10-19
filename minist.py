import math
import os
import torch
import torch.nn as nn
import cv2
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

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

# dim 为隐藏层的维度
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

# 定义ViT模型类
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        # 一个图片有多少个patch
        self.num_patches = (image_size // patch_size) ** 2
        # 每一个patch映射到一个vector
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, dim)
        # 可训练的位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        # 可训练的分类标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 本任务没必要同时用到编码器和解码器
        # self.transformer = nn.Transformer(dim, heads, depth)
        # 只用到depth层编码器 
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=256,batch_first=True)
        self.transformerencoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape # (N,3,32,32)
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size) # (N,3,8,4,8,4)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, self.patch_size * self.patch_size * 3) # (N,64,48)
        x = self.patch_embedding(x) # (N,64,64)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (N,1,64) class标签复制N份
        x = torch.cat((cls_tokens, x), dim=1) # (N,65,64) 拼接class标签 num_patches+1
        x += self.position_embedding # (N,65,64) 加上位置编码
        x = self.transformerencoder(x) # (N,65,64) 编码器输出
        # x = self.mlp_head(x[:, 0]) # (N,10)    x[:, 0] (N,64)只取第一个位置的输出 
        # x = torch.mean(x, dim=1)  # (N,64) # x[:, 0]可以替代为取所有位置的平均值
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x
    
# img = torch.ones(10, 3, 32, 32)
# model = ViT()
# x = model(img)
# print(x)


def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
  """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)