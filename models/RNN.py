import torch
from torch import nn
import utils
import models.support

csi_dataconvert = lambda x: x
csi_dataconvert.utils = utils


class RNN(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2, last_dim_size=64):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(last_dim_size, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        outputs = self.softmax(outputs)
        # outputs = self.softmax(outputs)
        return outputs


LEN_W = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader():
    import scipy
    import numpy as np
    from torch.utils.data import TensorDataset, random_split, DataLoader
    global LEN_W

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
    step_distance = 2
    for i in range(0, len(m2) - LEN_W, step_distance):
        data.append(m2[i:i + LEN_W])
        labels.append(0)
    for i in range(0, len(m4) - LEN_W, step_distance):
        data.append(m4[i:i + LEN_W])
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

    return train_dataloader, test_dataloader


def main():
    models.support.set_batch_size(256)
    train_dataloader, test_dataloader = get_dataloader()
    model = RNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    import Denoise
    denoiser = Denoise.get_trained_denoiser().to(DEVICE)
    for epoch in range(100):
        model.train()
        total_loss = 0
        for data, label in train_dataloader:
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            data = denoiser(data)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch},loss:{total_loss}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_dataloader):
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print(f"epoch:{epoch},acc:{correct / total}")


if __name__ == '__main__':
    main()
