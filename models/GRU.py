import numpy as np
import torch.nn
from sklearn.metrics import confusion_matrix
from torch import nn

# import phase_denoise
from . import support

LEN_W = support.LEN_W
DEVICE = support.DEVICE


class GRU(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=4,last_dim_size=64):
        super(GRU, self).__init__()
        self.gru = nn.GRU(last_dim_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2 * LEN_W, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out.reshape(out.shape[0], -1)
        out = self.drop(out)
        out = self.fc(out)
        return self.softmax(out)


def test(model, dataloader, denoiser=None):
    from sklearn.metrics import confusion_matrix
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            inputs = denoiser(inputs) if denoiser is not None else inputs

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))


def main():
    def preprocess(signal):
        ret = np.zeros(signal.shape)
        for j in range(signal.shape[0]):
            pass
            # ret[j] = torch.tensor(phase_denoise.corretct(signal[j]))
        return ret

    preprocess = None
    support.set_device(torch.device("cpu"))
    train_loader, test_loader = support.get_dataloader(domain=0, preprocess=preprocess)

    model = GRU(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    path = "../saved/gru_model.pth"
    save = support.get_default_save(path)
    # model = support.try_load_model(model, path) # 加载模型

    import Denoise
    # denoiser = Denoise.get_trained_denoiser().to(device=DEVICE)
    denoiserBuilder = Denoise.get_mlp_denoiser().to(device=DEVICE)
    denoiser = denoiserBuilder.denoiser
    denoiser = None
    l2_lambda = 0.01
    for epoch in range(10):
        model.train()
        total_loss = 0
        for data in train_loader:

            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # raw_inputs = inputs.to(DEVICE)
            # inputs = denoiser(inputs) if denoiser is not None else inputs

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # loss += denoiserBuilder(raw_inputs, labels)

            # 正则化避免过拟合
            l2_reg = torch.tensor(0.).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        save(model, total_loss)
        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, total_loss))

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                inputs = denoiser(inputs) if denoiser is not None else inputs

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            cm = confusion_matrix(all_labels, all_preds)
            print(cm)
            print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

    # model = support.load_model(model, path)
    other_loader = support.get_dataloader(domain=1, split=False, preprocess=preprocess)
    test(model, other_loader, denoiser)


if __name__ == '__main__':
    main()
    # model = GRU(num_classes=4).to(DEVICE)
    # path = "../saved/gru_model.pth"
    # model = support.try_load_model(model, path)  # 加载模型
    #
    # m2 = [csi_dataconvert.utils.CSI.get_csi_vector_from_packet(x) for x in
    #       csi_dataconvert.utils.read_udp_data_txt_to_bytes("../data/my_2m.txt")]
    # m2 = np.where(np.isnan(m2), 0, m2)
    # m2 = m2.real
    # m4 = [csi_dataconvert.utils.CSI.get_csi_vector_from_packet(x) for x in
    #       csi_dataconvert.utils.read_udp_data_txt_to_bytes("../data/my_4m.txt")]
    # m4 = np.where(np.isnan(m4), 0, m4)
    # m4 = m4.real

    # data = []
    # labels = []
    # for i in range(0, len(m2) - LEN_W, 2):
    #     data.append(m2[i:i + LEN_W])
    #     labels.append(0)
    # for i in range(0, len(m4) - LEN_W, 2):
    #     data.append(m4[i:i + LEN_W])
    #     labels.append(1)
    # data = [np.apply_along_axis(scipy.signal.medfilt, 1, x, kernel_size=5) for x in data]
    # data = torch.tensor(data, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.long)
    # dataset = torch.utils.data.TensorDataset(data, labels)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    #
    # model.eval()
    # cor = 0
    # for data in test_loader:
    #     inputs, labels = data
    #     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    #     outputs = model(inputs)
    #     _, predicted = torch.max(outputs.data, 1)
    #     cor += (predicted==labels).sum().item()
    # print("corr:",cor/len(dataset))
