import torch.nn
import support
from torch.utils.data import DataLoader


class PrototypicalNet(torch.nn.Module):
    def __init__(self, encoder, num_classes):
        super(PrototypicalNet, self).__init__()
        self.num_classes = num_classes
        self.prototypes = None

        self.encoder = encoder
        self.dist = torch.nn.PairwiseDistance(p=2)

    def forward(self, x):
        # x.shape = [batch_size, *]
        x = self.encoder(x)
        x = torch.cdist(x.unsqueeze(1), self.prototypes).mean(dim=[-2, -1])
        x = torch.softmax(x, dim=-1)
        return x

    def compute_prototypes(self, support, support_label):
        assert support_label < self.num_classes

        # 输入格式为[n_support, *]
        ck = self.encoder(support).sum(dim=0) / self.num_classes
        if self.prototypes is None:
            shape = [self.num_classes]
            for i in range(len(ck.shape)):
                shape.append(ck.shape[i])
            self.prototypes = torch.zeros(shape)
        self.prototypes[support_label] = ck.detach()


def random_sample(data_loader: DataLoader, class_id, n_support):
    class_data = []
    for data, labels in data_loader:
        data = data.to(support.DEVICE)
        data = data[labels == class_id]
        if len(data) == 0:
            continue
        class_data.append(data)
        if len(class_data) >= n_support:
            break
    class_data = torch.cat(class_data)
    class_data = class_data[:n_support]
    indices = torch.randperm(class_data.size(0))
    return class_data[indices]


Nc = 4
Ns = 10
Nq = 10
# Ns 20, Nq 20 => 0.3
# Ns 100, Nq 100 => 0.4

def pre_train(model: PrototypicalNet, data_loader):
    Sk = []
    Qk = []
    for k in range(Nc):
        sample = random_sample(data_loader, k, Ns + Nq)
        Sk.append(sample[:Ns])
        Qk.append(sample[Ns:])
    Sk = torch.stack(Sk)
    Qk = torch.stack(Qk)
    # 计算原型
    for k in range(Nc):
        model.compute_prototypes(Sk[k], k)
    return Sk, Qk


def train(model: PrototypicalNet, data_loader,Qk=None):
    if Qk is None:
        _, Qk = pre_train(model, data_loader)
    # 训练
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    total_loss = 0
    for k in range(Nc):
        opt.zero_grad()
        labels = torch.tensor([k] * Nq)
        output = model(Qk[k])
        loss = criterion(output, labels)
        total_loss += loss.item()
        loss.backward()
        opt.step()
    return total_loss / Nc

class SimpleGRU(torch.nn.Module):
    def __init__(self):
        super(SimpleGRU, self).__init__()
        self.gru = torch.nn.GRU(64, 64, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(64, 4)
    def forward(self,x):
        x,_=self.gru(x)
        return x


def main():
    support.set_device(torch.device("cpu"))
    # encoder = torch.nn.Conv1d(10, 2, 3)
    encoder = SimpleGRU()
    model = PrototypicalNet(encoder, 4).to(support.DEVICE)
    train_loader, valid_loader = support.get_dataloader(domain=-1)
    _,Qk = pre_train(model, train_loader)
    for i in range(100):
        loss = train(model, train_loader,Qk)
        print(f"epoch:{i},loss:{loss}")
        with torch.no_grad():
            cor = 0
            for data, labels in valid_loader:
                data,labels = data.to(support.DEVICE),labels.to(support.DEVICE)
                output = model(data)
                output = torch.argmax(output, dim=-1)
                cor += (output == labels).sum().item()
            print(f"accuracy:{cor / len(valid_loader)}")

    dataloader = support.get_dataloader(domain=1, split=False)
    support.test(model, dataloader)
    pass


if __name__ == '__main__':
    main()
