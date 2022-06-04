import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# 6.1.1
max_iters = 150
# pick a batch size, learning rate
batch_size = 15
learning_rate = 0.003
hidden_size = 64

trainset_x = torch.from_numpy(train_x).type(torch.float)
trainset_y = torch.from_numpy(train_y).type(torch.LongTensor)
train_loader = DataLoader(TensorDataset(trainset_x, trainset_y), batch_size=batch_size, shuffle=True)

testset_x = torch.from_numpy(test_x).type(torch.float)
testset_y = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(testset_x, testset_y), batch_size=batch_size, shuffle=False)


class orig_Net(nn.Module):
    def __init__(self, input_dim, hid_size, output_dim):
        super(orig_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


simple_net = orig_Net(train_x.shape[1], hidden_size,train_y.shape[1])
optimizer = torch.optim.SGD(simple_net.parameters(), lr=learning_rate, momentum=0.9)

train_loss = []
train_acc = []
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for data in train_loader:
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        targets = torch.max(labels, 1)[1]

        pred = simple_net(inputs)
        loss = nn.functional.cross_entropy(pred, targets)
        total_loss += loss.item()
        preds = torch.max(pred, 1)[1]
        avg_acc += preds.eq(targets.data).cpu().sum().item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_acc = avg_acc / train_y.shape[0]
    train_loss.append(total_loss)
    train_acc.append(avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

# plot acc curve
plt.plot(range(len(train_acc)), train_acc)
plt.xlabel("epoch")
plt.ylabel("average accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

# plot loss curve
plt.plot(range(len(train_loss)), train_loss)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, None)
plt.grid()
plt.show()


# 6.1.2
max_iters = 50
# pick a batch size, learning rate
batch_size = 15
learning_rate = 0.003

train_x = np.array([train_x[i, :].reshape((32, 32)) for i in range(train_x.shape[0])])
test_x = np.array([test_x[i, :].reshape((32, 32)) for i in range(test_x.shape[0])])

trainset_x, trainset_y = torch.from_numpy(train_x).type(torch.float32).unsqueeze(1), torch.from_numpy(train_y).type(torch.long)
test_x, test_y = torch.from_numpy(test_x).type(torch.float32).unsqueeze(1), torch.from_numpy(test_y).type(torch.long)

train_loader = DataLoader(TensorDataset(trainset_x, trainset_y), batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(test_x, test_y), shuffle=False)


class ConvNet1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(5 * 5 * 16, 50)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 5*5*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = ConvNet1(train_x.shape[1], train_y.shape[1])
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

train_loss = []
train_acc = []
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0

    for data in train_loader:

        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        targets = torch.max(labels, 1)[1]

        pred = net(inputs)
        loss = F.cross_entropy(pred, targets)
        total_loss += loss.item()
        preds = torch.max(pred, 1)[1]
        avg_acc += preds.eq(targets.data).cpu().sum().item() / labels.shape[0]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_acc = avg_acc / len(train_loader)
    train_loss.append(total_loss)
    train_acc.append(avg_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

# plot acc curve
plt.plot(range(len(train_acc)), train_acc)
plt.xlabel("epoch")
plt.ylabel("average accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

# plot loss curve
plt.plot(range(len(train_loss)), train_loss)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, None)
plt.grid()
plt.show()


# 6.1.3
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':      # only for 6.1.3, because of multiprocess calculation
    max_iters = 50
    # pick a batch size, learning rate
    batch_size = 15
    learning_rate = 0.003

    transformer_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transformer_test = transformer_train

    train_loader = DataLoader(
        torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=transformer_train),
        batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(
        torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=transformer_test),
        shuffle=False, num_workers=4)

    net = ConvNet2()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_loss = []
    train_acc = []
    for itr in range(max_iters):
        total_loss = 0
        avg_acc = 0

        for data in train_loader:

            inputs = torch.autograd.Variable(data[0])
            labels = torch.autograd.Variable(data[1])

            pred = net(inputs)
            loss = F.cross_entropy(pred, labels)
            total_loss += loss.item()
            preds = torch.max(pred, 1)[1]
            avg_acc += preds.eq(labels.data).cpu().sum().item() / labels.shape[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_acc = avg_acc / len(train_loader)
        train_loss.append(total_loss)
        train_acc.append(avg_acc)

        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

    # plot acc curve
    plt.plot(range(len(train_acc)), train_acc)
    plt.xlabel("epoch")
    plt.ylabel("average accuracy")
    plt.xlim(0, len(train_acc)-1)
    plt.ylim(0, None)
    plt.grid()
    plt.show()

    # plot loss curve
    plt.plot(range(len(train_loss)), train_loss)
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.xlim(0, len(train_loss)-1)
    plt.ylim(0, None)
    plt.grid()
    plt.show()
