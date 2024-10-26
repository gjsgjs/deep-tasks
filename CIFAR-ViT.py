import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from minist import ViT, classBalance_loss, get_cosine_schedule_with_warmup
from qqdm import qqdm
import time
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# 数据目录
train_data_dir = 'CIFAR10_imbalanced'
test_data_dir = 'CIFAR10_balance'

  # 必要目录
log_dir = os.path.join(".", 'vit_run')
ckpt_dir = os.path.join(".", 'vit_checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)


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

from torchvision.models import resnet18

def train():
    # 
    writer = SummaryWriter(f'vit_run/{args.model}_run')
    #
    batch_size = args.batch_size
    warmup_steps = args.warmup_steps
    total_steps = args.total_steps
    learning_rate = args.learning_rate
    # 加载训练数据集
    train_dataset = datasets.ImageFolder(train_data_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据集
    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 训练模型并评估性能
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型替换
    model = ViT(model_type=args.model).to(device)
    if args.model == 'resnet18':
        model = resnet18(pretrained=False, num_classes=10).to(device)
    print("加载的模型是："+args.model)
    # 计算类权重,加权损失函数 写一个args
    class_counts = torch.tensor([train_dataset.targets.count(i) for i in range(10)])
    class_weights = 1. / class_counts
    class_weights = class_weights.to(device)
    # 定义损失函数和优化器
    if args.weight:
        if args.classifier_loss !='CE':
            print('Using '+args.classifier_loss+' weighted loss')
        print('Using weighted loss')
        criterion = nn.CrossEntropyLoss(class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 训练模型
    num_epochs = args.epochs
    step = 0
    for e,epoch in enumerate(range(num_epochs)):
        progress_bar = qqdm(train_loader)
        model.train()
        train_loss = 0.0
        start_time = time.time()  # 记录开始时间
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if args.classifier_loss == 'CE':
                loss = criterion(outputs, labels)
            else :
                loss = classBalance_loss(device, labels, outputs, class_counts, 10, args.classifier_loss, 0.999)
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
        end_time = time.time()  # 记录结束时间
        epoch_duration = end_time - start_time  # 计算持续时间
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}')
        writer.add_scalar('Time/epoch', epoch_duration, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], step)
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)

        if step>=total_steps:
            print('Training finished')
            break
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
                # loss = criterion(outputs, labels)
                if args.classifier_loss == 'CE':
                    loss = criterion(outputs, labels)
                else :
                    loss = classBalance_loss(device, labels, outputs, class_counts, 10, args.classifier_loss, 0.999)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Test Loss: {test_loss/len(test_loader)}, Accuracy: {100 * correct / total}%')
        writer.add_scalar('Loss/test', test_loss/len(test_loader), epoch)
        writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)

        if (epoch+1) % args.save_epoch == 0:
            train_loss_avg = train_loss / len(train_loader)
            test_loss_avg = test_loss / len(test_loader)
            accuracy = 100 * correct / total
            torch.save(model.state_dict(), f'vit_checkpoints/{args.model}_epoch_{epoch+1}_trainloss_{train_loss_avg:.2f}_testloss_{test_loss_avg:.2f}_acc_{accuracy:.2f}.pth')
            print(f'Models saved at epoch {epoch},step {step}')

def infer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(model_type=args.model).to(device)
    summary(model, (3, 32, 32))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VIT model.')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'conformer','lsa','resnet18'], help='which model to use: transformer or confermer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size for training')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='warmup steps for learning rate scheduler')
    parser.add_argument('--total_steps', type=int, default=7500, help='total steps for learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=10, help='save model every save_epoch')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help='Mode to run the script in: train or infer')
    parser.add_argument('--weight',action='store_false', help='Whether to weight the loss function')
    parser.add_argument('--classifier_loss', type=str, default='sigmoid', choices=['CE', 'softmax', 'sigmoid', 'focal'], help='classifier loss function')
    args = parser.parse_args()

    if args.mode == 'train':
        print('预计训练'+str(args.epochs)+'轮')
        print("预计训练"+str(args.total_steps)+"步")
        train()
    elif args.mode == 'infer':
        infer()