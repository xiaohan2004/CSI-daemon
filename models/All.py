import abc
import os

import numpy as np
import torch.optim
import pywt

# import models
from . import support


class Model(abc.ABC):
    @abc.abstractmethod
    def get_model(self): ...

    def fit(
        self,
        train_loader,
        valid_loader=None,
        epoch_size=5,
        optimizer=None,
        criterion=None,
        lr=1e-3,
        howtoprint=None,
        alpha=0.01,
        bad_count_limit=20,
    ):
        model = self.get_model().to(support.DEVICE)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        if howtoprint is None:

            def dummy_print(*args, **kwargs):
                pass

            howtoprint = print

        # Initialize the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        bad_count = 0
        # bad_count_limit = 6
        min_total_loss = 1e9
        last_acc = None
        last_valid_acc = None
        for epoch in range(epoch_size):
            # print(f"Epoch:{epoch + 1}")
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(support.DEVICE), labels.to(support.DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss += getattr(outputs, "e_loss", lambda _, __: 0)(inputs, labels)
                # 对损失进行正则化，避免过拟合
                nn_loss = torch.tensor(0.0).to(support.DEVICE)
                nn_total = 0
                for param in model.parameters():
                    nn_total += 1
                    nn_loss += alpha * torch.norm(param, 1)
                loss += nn_loss / nn_total
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            howtoprint(
                f"Epoch:{epoch + 1}, Loss:{total_loss}, Train Accuracy:{correct / total}"
            )
            last_acc = correct / total
            # stop when loss is not decreasing

            if total_loss < min_total_loss:
                min_total_loss = total_loss
                bad_count = 0
                torch.save(model.state_dict(), "/tmp/best_model.pth")
            elif min_total_loss <= total_loss:
                bad_count += 1
            if bad_count > bad_count_limit:
                howtoprint("Early stop")
                model.load_state_dict(torch.load("/tmp/best_model.pth"))
                os.remove("/tmp/best_model.pth")
                break

            # Update the learning rate
            scheduler.step(total_loss)

            # 验证
            if valid_loader is None:
                continue
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                all_preds = []
                all_labels = []
                for data in valid_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(support.DEVICE), labels.to(
                        support.DEVICE
                    )

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                howtoprint(f"Valid Accuracy: {correct / total}")
                last_valid_acc = correct / total
        return last_acc, last_valid_acc

    def test(self, test_loader):

        from sklearn.metrics import confusion_matrix

        model = self.get_model().to(support.DEVICE)
        model.eval()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(support.DEVICE), labels.to(support.DEVICE)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            print(f"Test Accuracy: {correct / total}")
            test_acc = correct / total
        cm = confusion_matrix(all_labels, all_preds)
        return cm, test_acc


class SimpleMLP(Model):
    def __init__(self, num_classes=4, last_dim_size=64):
        super().__init__()
        self.out_feature = num_classes
        self.last_dim_size = last_dim_size
        from torch import nn

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.last_dim_size * support.LEN_W, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_feature),
            nn.Softmax(),
        )

    def get_model(self):
        return self.model


class SimpleRNN(Model):
    def __init__(self, num_classes=4, hidden_dim=64, last_dim_size=64):
        super().__init__()
        from .RNN import RNN

        self.model = RNN(
            num_classes=num_classes, hidden_dim=hidden_dim, last_dim_size=last_dim_size
        )

    def get_model(self):
        return self.model


class SimpleGRU(Model):
    def __init__(self, hidden_dim=64, num_classes=4, last_dim_size=64):
        super().__init__()
        from .GRU import GRU

        self.model = GRU(
            last_dim_size=last_dim_size, hidden_dim=hidden_dim, num_classes=num_classes
        )

    def get_model(self):
        return self.model


class SimpleLSTM(Model):
    def __init__(self, in_features, classes_num=4, hidden_dim=64):
        super().__init__()
        from .LSTM import LSTM

        self.model = LSTM(
            in_features=in_features, classes_num=classes_num, hidden_dim=hidden_dim
        )

    def get_model(self):
        return self.model


class SimpleBiLSTM(Model):
    def __init__(self, in_features, classes_num=4, hidden_dim=64):
        super().__init__()
        from .LSTM import BiLSTM

        self.model = BiLSTM(
            in_features=in_features, classes_num=classes_num, hidden_dim=hidden_dim
        )

    def get_model(self):
        return self.model


class SimpleLeNet(Model):
    def __init__(self, num_classes=4):
        super().__init__()
        self.out_feature = num_classes
        from torch import nn

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 2), padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # nn.Conv2d(64, 96, (3, 3), stride=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 2 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_feature),
            nn.Softmax(),
        )

    def get_model(self):
        return self.model


class SimpleResNet18(Model):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        from .ResNet import resnet18

        self.model = resnet18(in_features, num_classes)

    def get_model(self):
        return self.model


class SimpleResNet34(Model):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        from .ResNet import resnet34

        self.model = resnet34(in_features, num_classes)

    def get_model(self):
        return self.model


class SimpleResNet50(Model):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        from .ResNet import resnet50

        self.model = resnet50(in_features, num_classes)

    def get_model(self):
        return self.model
