import numpy as np
import torch

import support_web
from models import support, All


class CNN_GRU(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_classes=4, last_dim_size=64):
        super(CNN_GRU, self).__init__()
        self.cnn = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        self.norm = torch.nn.BatchNorm2d(1)
        self.a1 = torch.nn.ReLU()
        self.gru = torch.nn.GRU(last_dim_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.drop = torch.nn.Dropout(0.2)
        self.rnn = torch.nn.RNN(last_dim_size, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.a2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.a3 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.rnn(x)[1]
        x = self.flatten(x[-1])
        x = self.fc2(x)
        return self.a3(x)
        # x1 = x[:]
        # x1[:, 1:] = x[:, :-1]
        # x1 = x1 - x
        # x1 = torch.stack([x1, x], dim=1)
        # x1 = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        # x1 = self.cnn(x1)
        # x1 = self.norm(x1)
        # x1 = self.a1(x1)
        # x1 = x1.reshape(x1.shape[0], x1.shape[-2], x1.shape[-1])
        # x1 = self.rnn(x)[0]
        # x1 = self.flatten(x1)
        # x1 = self.fc2(x1)
        # x1 = self.a3(x1)
        # return x1


if __name__ == '__main__':
    model = CNN_GRU(num_classes=2)
    # model = All.SimpleRNN(num_classes=2).get_model()
    csvs = (["../saved_csv/read_from_serial_2024-11-11_19-59-52.csv",
             "../saved_csv/read_from_serial_2024-11-11_10-42-52.csv"],
            [1, 0])
    train, valid = support_web.get_dataloader_from_csv(csvs, split=True, preprocess=lambda x: np.abs(x),batch_size=256)
    opt = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        right = 0
        total = 0
        total_loss = 0
        for (x, y) in train:
            py = model(x)
            # print(f"py shape{py.shape}")
            # print(f"y shape{y.shape}")
            opt.zero_grad()
            l = loss(py, y)
            l.backward()
            opt.step()
            right += (y == torch.argmax(py, dim=1)).sum()
            total += len(y)
            total_loss += l.item()
        acc = right / total
        print(f"epoch:{epoch},total_loss:{total_loss},total_acc:{acc}")
