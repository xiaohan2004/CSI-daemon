import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from . import support

import utils


class LeNet(nn.Module):
    def __init__(self,num_classes=2):
        super(LeNet, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,LEN_W,64)
            nn.Conv2d(1, 32, 5,padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 2), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # nn.Conv2d(64, 96, (3, 3), stride=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*2*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = x.view(-1, 64*2*8)
        out = self.fc(x)
        return out


LEN_W = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader():
    len_w = LEN_W

    # 只要振幅，以及nan设置为0
    m2 = [csi_dataconvert.utils.CSI.get_csi_vector_from_packet(x) for x in
          csi_dataconvert.utils.read_udp_data_txt_to_bytes("../0510/2M.txt")]
    m2 = np.where(np.isnan(m2), 0, m2)
    m2 = m2.real
    m4 = [csi_dataconvert.utils.CSI.get_csi_vector_from_packet(x) for x in
          csi_dataconvert.utils.read_udp_data_txt_to_bytes("../0510/4M.txt")]
    m4 = np.where(np.isnan(m4), 0, m4)
    m4 = m4.real

    data = []
    labels = []
    step_distance=2
    for i in range(0,len(m2) - len_w,step_distance):
        data.append(m2[i:i + len_w])
        labels.append(0)
    for i in range(0,len(m4) - len_w,step_distance):
        data.append(m4[i:i + len_w])
        labels.append(1)

    # decrease noise
    data = [np.apply_along_axis(scipy.signal.medfilt, 1, x, kernel_size=5) for x in data]

    data = torch.tensor(data, dtype=torch.float32)
    print(data.shape)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader,test_dataloader

def train(model,trainloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    for data, target in trainloader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("avg_loss:",total_loss/len(trainloader))
    return total_loss

def valid(model, valide_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in valide_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print('Accuracy of the network on the test data: %d %%' % (
                100 * correct / total))


def main():
    model = LeNet().to(DEVICE)

    train_loader, valid_loader = get_dataloader()
    last_loss = None
    for epoch in range(300):
        total_loss = train(model,train_loader)
        if last_loss is None or last_loss > total_loss:
            last_loss = total_loss
            torch.save(model.state_dict(), "../saved/LeNet_model.pth")

    model.load_state_dict(torch.load("../saved/LeNet_model.pth"))
    valid(model,valid_loader)


if __name__ == '__main__':
    main()
