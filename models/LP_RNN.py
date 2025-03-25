import torch
from torch import nn

from models import support


class LP(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2, L=support.LEN_W // support.STEP_DISTANCE, H=64):
        super(LP, self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones(L, H))
        self.bias = nn.parameter.Parameter(torch.zeros(L, H))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x * self.weight + self.bias
        x = self.activation(x)
        return x


class LP_RNN(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2, last_dim_size=64, L=support.LEN_W // support.STEP_DISTANCE):
        super().__init__()
        self.lp = LP(L, last_dim_size)
        self.rnn = nn.RNN(last_dim_size, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.lp(x)
        lp_loss = 0.01 * torch.norm(torch.mean(x, dim=(1, 2)), p=1)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        outputs = self.softmax(outputs)
        outputs.e_loss = lambda _, labels: torch.where(labels == 0, lp_loss, 0).sum()
        return outputs


from .All import Model


class SimpleLP_RNN(Model):
    def __init__(self, last_dim_size, num_classes, L=support.LEN_W // support.STEP_DISTANCE):
        self.model = LP_RNN(last_dim_size=last_dim_size, num_classes=num_classes, L=L)

    def get_model(self):
        return self.model
