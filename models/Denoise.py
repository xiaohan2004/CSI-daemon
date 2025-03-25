import math

import torch
from torch import nn

import models.support
import  matplotlib.pyplot as plt

class SignalSimilarity(nn.Module):
    """
    用于判断信号和历史信号的相似度，相似度越高，loss越小
    """

    def __init__(self, back_signal=None):
        super().__init__()
        self.back_signal = back_signal
        self.back_signal_len = 0
        self.eval_ret = lambda x, y: torch.mean((x - y) ** 2).sum()

    def forward(self, x):
        if self.back_signal is None:
            ret = self.eval_ret(x, 0)
            self.back_signal = x.detach()
        else:
            ret = self.eval_ret(x, self.back_signal)
            x = x.detach()
            self.back_signal = (self.back_signal_len * self.back_signal + x) / (self.back_signal_len + 1)

        self.back_signal += 1
        return ret

    def clear(self):
        self.back_signal_len = 1


class SignalDifferenceLoss(nn.Module):
    """
    用于判断两个信号的差异度，差异度越大，loss越小
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if x is None or y is None:
            return torch.tensor(torch.inf)

        if x is None:
            x = y
            y = torch.zeros(x.shape)
        if y is None:
            y = torch.zeros(x.shape)

        return torch.exp(-torch.abs(x - y)).sum()


class SignalSmoothLoss(nn.Module):
    """
    用于判断信号的平滑度，平滑度越大，loss越小
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.std(torch.diff(x, 2))


class DenoiseBuilder(nn.Module):
    def __init__(self, denoiser, num_classes):
        super().__init__()
        self.denoiser = denoiser
        self.noise_similarity = SignalSimilarity()
        self.signal_similarity = []
        for i in range(num_classes):
            self.signal_similarity.append(SignalSimilarity())
        self.diff_loss = SignalDifferenceLoss()
        self.signal_smooth = SignalSmoothLoss()

    def inner_forward(self, x, label):
        assert label < len(self.signal_similarity)
        # 提取噪声
        noise = self.denoiser(x)
        # 去噪后信号
        signal = x - noise

        # 计算噪声相似度
        # loss1 = self.noise_similarity(noise)
        loss1 = torch.tensor(0.0)

        # 计算信号相似度
        loss2 = self.signal_similarity[label](signal)

        # 计算信号差异度
        loss3 = torch.tensor(0.0)
        for i in range(len(self.signal_similarity)):
            if i != label:
                loss3 += self.diff_loss(signal, self.signal_similarity[i].back_signal)
        # 计算平滑度
        loss4 = self.signal_smooth(signal)

        return loss1 + 0.2*loss2 + loss3 + 0.2*loss4

    def forward(self, x, labels):
        loss = torch.tensor(0.0)
        for i in range(len(labels)):
            loss += self.inner_forward(x[i], labels[i])
        return loss


def get_mlp_denoiser():
    mlp = nn.Sequential(
        nn.Linear(64, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
    )
    denoiseBuilder = DenoiseBuilder(mlp, 4)
    return denoiseBuilder

def get_trained_denoiser():
    mlp = nn.Sequential(
        nn.Linear(64, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
    )
    mlp.load_state_dict(torch.load("../saved/denoise.pth"))
    for param in mlp.parameters():
        param.requires_grad = False
    mlp.eval()
    return mlp

def main():
    models.support.set_len_w(200)
    models.support.set_device(torch.device("cpu"))
    device = models.support.DEVICE
    mlp = nn.Sequential(
        nn.Linear(64, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
    )
    denoiseBuilder = DenoiseBuilder(mlp, 4).to(device)
    opt = torch.optim.SGD(denoiseBuilder.parameters(), lr=0.001)

    train_loader, _ = models.support.get_dataloader(step_distance=10)
    for epoch in range(500):
        total_loss = 0.0
        min_loss = None
        print(f"Epoch {epoch}")
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            loss = denoiseBuilder(data, labels)
            total_loss += loss.item()
            loss.backward()
            opt.step()
            # print(loss.item())
        print(f"total_loss: {total_loss}")
        if min_loss is None or (min_loss > total_loss):
            min_loss = total_loss
            torch.save(denoiseBuilder.denoiser.state_dict(), "../saved/denoise.pth")

    denoiser = denoiseBuilder.denoiser
    denoiser.load_state_dict(torch.load("../saved/denoise.pth"))
    denoiser.eval()
    test_loader = models.support.get_dataloader(domain=1, split=False,step_distance=10)
    for data, _ in test_loader:
        for s in data:
            noise = denoiser(s)
            signal = s - noise
            fig,axes = plt.subplots(3,1)
            axes[0].plot(range(len(s)),s.detach().numpy())
            axes[1].plot(range(len(noise)),noise.detach().numpy())
            axes[2].plot(range(len(signal)),signal.detach().numpy())
            plt.show()

if __name__ == '__main__':
    # main()
    denoiser = get_trained_denoiser()
    test_loader = models.support.get_dataloader(domain=1, split=False, step_distance=10)
    for data, _ in test_loader:
            noise = denoiser(data)
            signal = data - noise
            s = data[0]
            signal = signal[0]
            noise = noise[0]
            fig, axes = plt.subplots(3, 1)
            axes[0].plot(range(len(s)), s.detach().numpy())
            axes[1].plot(range(len(noise)), noise.detach().numpy())
            axes[2].plot(range(len(signal)), signal.detach().numpy())
            plt.show()