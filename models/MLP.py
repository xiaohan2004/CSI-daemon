import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
import torch.nn
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader

import matplotlib.pyplot as plt

import csi_dataconvert.utils
import phase_denoise


def decrease_noise(sig):
    (cA, cD) = pywt.dwt(sig, 'db1')
    max_limit = 100
    cA = np.where(abs(cA) < max_limit, 0, cA)
    cD = np.where(abs(cD) < max_limit, 0, cD)

    limit = 200
    cA = np.where(cA >limit, limit , cA)
    cA = np.where(cA<-limit, -limit, cA)
    cD = np.where(cD<-limit,-limit, cD)
    cD = np.where(cD >limit, limit, cD)
    sig = pywt.idwt(cA, cD, 'db1')
    return sig


def get_signal(file):
    packets = csi_dataconvert.utils.read_udp_data_txt_to_bytes(file)
    csi_vectors = [csi_dataconvert.utils.CSI.get_csi_vector_from_packet(x) for x in packets]
    return np.array(csi_vectors)


class MLP(torch.nn.Module):
    def __init__(self, len_w=10):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * len_w, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    len_w = 5

    # 只要振幅，以及nan设置为0
    m2 = get_signal("../0510/2M.txt")
    m2 = np.where(np.isnan(m2), 0, m2)
    # m2 = csi_dataconvert.utils.CSI.get_amplitude(m2)
    m2 = m2.imag
    m4 = get_signal("../0510/4M.txt")
    m4 = np.where(np.isnan(m4), 0, m4)
    # m4 = csi_dataconvert.utils.CSI.get_amplitude(m4)
    m4 = m4.imag

    # m2 = np.where(m2>1e6, 1e6, m2)
    # plt.plot(range(len(m2)), m2)
    # plt.show()
    # plt.plot(range(len(m2)), decrease_noise(m2)[:,19])
    # plt.show()
    # filtered_m2 = np.apply_along_axis(scipy.signal.medfilt, 1, m2, kernel_size=5)
    # plt.plot(range(len(filtered_m2)), filtered_m2)
    # plt.show()

    data = []
    labels = []
    for i in range(len(m2) - len_w):
        data.append(m2[i:i + len_w])
        labels.append(0)
    for i in range(len(m4) - len_w):
        data.append(m4[i:i + len_w])
        labels.append(1)

    # decrease noise
    data = [np.apply_along_axis(scipy.signal.medfilt, 1, x, kernel_size=5) for x in data]
    # data = np.array(data)
    # for i in range(len(data)):
    #     for j in range(data.shape[1]):
    #         data[i,j] = torch.tensor(phase_denoise.corretct(data[i,j]))

    data = torch.tensor(data, dtype=torch.float32)
    print(data.shape)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data, labels)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    model = MLP(len_w).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            print(f"y shape:{y.shape}")
            optimizer.zero_grad()

            x = model(x)
            loss = criterion(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch}, loss {total_loss}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            x = model(x)
            pred = torch.argmax(x, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        print(f"total:{total},correct:{correct},Accuracy: {correct / total}")


if __name__ == "__main__":
    main()
