from . import support
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_features, classes_num=4, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim * support.LEN_W, classes_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return self.softmax(out)


class BiLSTM(nn.Module):
    def __init__(self, in_features, classes_num=4, hidden_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 * support.LEN_W, classes_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return self.softmax(out)


def main():
    train_loader, test_loader = support.get_dataloader()

    model = BiLSTM(64, 4).to(support.DEVICE)
    # support.simple_train(model, train_loader, test_loader, 1000)
    support.catch_cancel(lambda: support.l2_regex_train(model, 0.1, train_loader, test_loader, ))
    test_loader = support.get_dataloader(domain=1, split=False)
    support.test(model, test_loader)


if __name__ == '__main__':
    main()
