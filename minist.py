import math
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

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

# my_attention
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q, k, v = (t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool) # 生成一个对角线为True的mask
        mask_value = -torch.finfo(dots.dtype).max #  获取负无穷大值
        dots = dots.masked_fill(mask, mask_value) # 应用掩码，将对角线元素设置为负无穷大 消除自己的相关性

        attn = self.attend(dots) # 计算注意力权重(对角线为0)

        out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        return self.to_out(out)

# seq = torch.ones(10, 20, 100)
# lsa = LSA(100)
# x = lsa(seq)



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # 每一个heads的qkv(1)维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1) # 注意力权重计算
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # 一个dim到heads*qkv(3)的映射

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() # 将多头的输出映射到dim维度

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 拆分qkv 3个(N,LEN,dim_head*heads)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # qkv每一个 (N,heads,LEN,dim_head)
        q, k, v = (t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # 点积计算attention权重

        attn = self.attend(dots) # attention经过一个softmax

        out = torch.matmul(attn, v) # 根据权重乘以v得到输出
        # out = rearrange(out, 'b h n d -> b n (h d)') # 将heads维度合并
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], -1)
        return self.to_out(out) # 输出映射回dim维度

class myTransformerencoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head,num_layers,dim_feedforward=256, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, dim_feedforward, dropout=dropout),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for attention, feed_forward, norm1, norm2 in self.layers:
            x = norm1(x + (attention(x)))
            x = norm2(x + (feed_forward(x)))
            # 先norm
            # x = x + self.dropout(self.attention(self.norm1(x)))
            # x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x
    
class mylsaTransformerencoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head,num_layers,dim_feedforward=256, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LSA(dim, heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, dim_feedforward, dropout=dropout),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for attention, feed_forward, norm1, norm2 in self.layers:
            x = norm1(x + (attention(x)))
            x = norm2(x + (feed_forward(x)))
            # 先norm
            # x = x + self.dropout(self.attention(self.norm1(x)))
            # x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ConvModule(nn.Module):
     def __init__(self, dim, kernel_size=31, dropout=0.):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

     def forward(self, x):
        # Apply LayerNorm
        x = self.layer_norm(x)
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # (N, L, d) -> (N, d, L)
        # Pointwise Convolution 1 + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        # Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
         # Pointwise Conv2
        x = self.pointwise_conv2(x)
        # Transpose back
        x = x.transpose(1, 2)   # (N, d, L) -> (N, L, d)
        x = self.dropout(x)
        return x
     
# seq = torch.ones(10, 20, 64)  # (batch_size, seq_len, dim)
# conv_module = ConvModule(dim=64)
# output = conv_module(seq)  

class myConformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, num_layers, dim_feedforward=256, kernel_size=31, dropout=0.,half_step_residual=True):
        super().__init__()
        # 是否使用半步残差
        if half_step_residual:
            self.residual_factor = 0.5
        else:
            self.residual_factor = 1

        self.layers = nn.ModuleList([
            nn.ModuleList([
                FeedForward(dim, dim_feedforward, dropout=dropout),
                Attention(dim, heads, dim_head=dim_head, dropout=dropout),
                ConvModule(dim, kernel_size=kernel_size, dropout=dropout),
                FeedForward(dim, dim_feedforward, dropout=dropout),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]) for _ in range(num_layers)
        ])

    def forward(self, x):
        for feed_forward1, attention, conv_module, feed_forward2, norm1, norm2, norm3,norm4 in self.layers:
            x = norm1(x + self.residual_factor * feed_forward1(x))
            x = norm2(x +  attention(x))
            x = norm3(x +  conv_module(x))
            x = norm4(x + self.residual_factor * feed_forward2(x))
        return x
           
    
# seq = torch.ones(10, 20, 64) # (N,len,dim)
# tran = myConformerLayer(dim=64, heads=8,dim_head=64,num_layers=6)
# x = tran(seq)


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep




# 定义ViT模型类
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=256, depth=6, heads=8, mlp_dim=128,dropout=0.1, model_type='transformer'):
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
        if model_type == 'transformer':
            self.transformerencoder = myTransformerencoderLayer(dim, heads, dim_head=64, num_layers=depth,dropout=dropout)
        elif model_type == 'conformer':
            self.transformerencoder = myConformerLayer(dim, heads, dim_head=64, num_layers=depth,dropout=dropout)
        elif model_type == 'lsa':
            self.transformerencoder = mylsaTransformerencoderLayer(dim, heads, dim_head=64, num_layers=depth,dropout=dropout)
        # pytorch的这个怎么没有dim_head
        # encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=256,batch_first=True)
        # self.transformerencoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.attention_pooling = SelfAttentionPooling(dim) # 不知道有没有用


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
        # x = torch.mean(x, dim=1)
        x = self.attention_pooling(x)
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


import torch.nn.functional as F

def classBalance_loss(device, labels, logits, samples_per_cls, no_of_classes, loss_type, beta):
    '''Paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    
    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    params:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        samples_per_cls: A python list of size [no_of_classes].
        no_of_classes: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.
    Returns:
        cb_loss: A float tensor representing class balanced loss
    '''
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

 
    if loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss