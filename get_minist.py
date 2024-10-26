# coding: utf-8
import argparse
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def get_alldata():
    (x_train, t_train), (x_tests, t_tests) = load_mnist(normalize=True, one_hot_label=False, flatten=False)
    # (x_train, t_train), (x_tests, t_tests) = load_mnist(normalize=False, one_hot_label=False, flatten=False)
    return x_train, t_train, x_tests, t_tests


import torch
import cv2
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=(2, 2), stride=2)(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

 
    def infer(self, image, device):
        image = torch.Tensor(image).to(device)
        with torch.no_grad():
            output = self(image)
            _, predicted = torch.max(output, 1)
        return predicted.item()
    
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv = nn.Sequential(
#             # (BATCH_SIZE,1,28,28) -> (BATCH_SIZE,32,30,30)
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=2),
#             # 非线性层
#             nn.ReLU(),
#             # (BATCH_SIZE,32,30,30) -> (BATCH_SIZE,32,15,15)
#             nn.MaxPool2d(kernel_size=(2, 2)),
#             # (BATCH_SIZE,32,15,15) -> (BATCH_SIZE,64,16,16)
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=1, padding=2),
#             # 非线性层
#             nn.ReLU(),
#             # (BATCH_SIZE,64,16,16) -> (BATCH_SIZE,64,8,8)
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#         self.fc = nn.Linear(64 * 8 * 8, 10)

#     def forward(self, x):
#         x = self.conv(x)
#         flat = x.view(x.size(0), -1)
#         y = self.fc(flat)
#         return y




def infer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =  LeNet5().to(device)
    model.load_state_dict(torch.load("checkpoints\model_epoch_10.pth"))
    x_train, t_train, x_test, t_test = get_alldata()
    for x in range(5):
        img = x_train[x]
        i = model.infer(img, torch.device('cuda'))
        print(i)
        img = img.reshape(28, 28)
        img = (img * 255).astype(np.uint8)  # 将图像像素值从 [0, 1] 变为 [0, 255]
        cv2.imshow(f'Image {i+1}', img)
        cv2.waitKey(0)  # 等待按键按下

    cv2.destroyAllWindows() 
    
def train():
    writer = SummaryWriter('runs')
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据
    x_train, t_train, x_test, t_test = get_alldata()

    # np转tensor
    x_train = torch.Tensor(x_train).to(device)
    t_train = torch.Tensor(t_train).long().to(device)
    x_test = torch.Tensor(x_test).to(device)
    t_test = torch.Tensor(t_test).long().to(device)
    # 转换为 TensorDataset
    train_dataset = TensorDataset(x_train, t_train, )
    test_dataset = TensorDataset(x_test, t_test, )
    #
    batch_size = 64
    learning_rate = 1e-3
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    model = LeNet5().to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, target in train_loader:
            batch, target = batch.to(device), target.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            # print(batch_X[:, :25, :].shape)
            outputs = model(batch)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {total_loss / len(train_loader.dataset):.4f}')
        writer.add_scalar('Loss/train', loss, epoch)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, target in test_loader:
                batch, target = batch.to(device), target.to(device)  # 将数据移动到GPU
                outputs = model(batch)
                loss = criterion(outputs, target)
                test_loss += loss.item() * batch.size(0)

                # 对于分类问题，可以计算准确率
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Test Loss: {test_loss / total:.4f}')
        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        # 记录损失和准确率
        writer.add_scalar('Loss/test', test_loss / total, epoch)
        writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)

         # 保存模型参数
        torch.save(model.state_dict(), f'checkpoint/model_epoch_{epoch + 1}.pth')


    img = torch.ones(1, 1, 28, 28)
    model = LeNet5()
    writer.add_graph(model, img)
    writer.close()

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def test():
     # 获取文件夹中的所有图片文件
    input_folder = 'test_pics'
    output_size=(28, 28)
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    model.load_state_dict(torch.load("checkpoints\model_epoch_10.pth"))
    
    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Failed to load image {image_file}")
            continue
        
        # 裁剪图像到指定大小
        resized_image = cv2.resize(image, output_size)
        
        # 二值化处理
        _, binary_image = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY)

        # 反转图像
        binary_image = cv2.bitwise_not(binary_image)

        # 应用不同程度的平移和旋转
        if args.deal == 'translate':
            binary_image = translate(binary_image, 2, 2)
        elif args.deal == 'rotate':
            binary_image = rotate(binary_image, 30)
        # binary_image = translate(binary_image, 2, 2)
        # binary_image = rotate(binary_image, 15)
        
        # 可视化处理后的图像
        cv2.imshow('Processed Image', binary_image)
        cv2.waitKey(0)  # 等待按键按下
        cv2.destroyAllWindows()

        # 将图像转换为模型的输入
        img = binary_image / 255.0
        img = img.reshape(1, 1, 28, 28)
        i = model.infer(img, device)
        print(f'Predicted number is {i}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or infer LeNet-5 model.')
    #parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help='Mode to run the script in: train or infer')
    parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer','test'], help='Mode to run the script in: train or infer')
    parser.add_argument('--deal',type=str,default='none',choices=['none','translate','rotate'],help='deal with the image')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'infer':
        infer()
    elif args.mode == 'test':
        test()