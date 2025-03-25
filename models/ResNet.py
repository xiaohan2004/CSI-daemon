from torch import nn
from . import support

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, residual_path=False):
        super(BasicBlock, self).__init__()
        self.residual_path = residual_path
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm1d(out_channels)
        self.a1 = nn.ReLU()

        self.c2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.b2 = nn.BatchNorm1d(out_channels)
        self.a2 = nn.ReLU()

        if residual_path:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x.clone()
        x = self.a1(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))

        if self.residual_path:
            identity = self.bn(self.conv(identity))
        return self.a2(x + identity)


from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, residual_path=False):
        super(Bottleneck, self).__init__()
        self.residual_path = residual_path or in_channels != out_channels * self.expansion
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.c1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.b1 = nn.BatchNorm1d(out_channels)
        self.a1 = nn.ReLU()

        self.c2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.b2 = nn.BatchNorm1d(out_channels)
        self.a2 = nn.ReLU()

        self.c3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, stride=1)
        self.b3 = nn.BatchNorm1d(out_channels * 4)
        self.a3 = nn.ReLU()

        if self.residual_path:
            self.conv = nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, stride=stride)
            self.bn = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x):
        identity = x.clone()
        x = self.a1(self.b1(self.c1(x)))
        x = self.a2(self.b2(self.c2(x)))
        x = self.b3(self.c3(x))
        if self.residual_path:
            identity = self.bn(self.conv(identity))
        return self.a3(x + identity)


class ResNet(nn.Module):
    def __init__(self, in_channels, Block, block_list, classes_num=1000):
        super(ResNet, self).__init__()
        self.c1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2)
        self.b1 = nn.BatchNorm1d(64)
        self.a1 = nn.ReLU()
        self.p1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.blocks = nn.Sequential()
        last_out = 64
        out = 64
        for block_id in range(len(block_list)):
            out = 64 * 2 ** block_id
            for layer_id in range(block_list[block_id]):
                if block_id == 0 and layer_id == 0:
                    block = Block(last_out, out, residual_path=False)
                    last_out = out*Block.expansion
                elif block_id != 0 and layer_id == 0:
                    block = Block(last_out, out, stride=2, residual_path=True)
                    last_out = out*Block.expansion
                else:
                    block = Block(last_out, out, residual_path=False)
                self.blocks.append(block)
        self.p2 = nn.AdaptiveAvgPool1d(1)
        self.d = nn.Linear(out * Block.expansion, classes_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.p1(self.a1(self.b1(self.c1(x))))
        x = self.blocks(x)
        x = self.p2(x)
        x = x.reshape(x.shape[0], -1)

        x = self.d(x)
        return self.softmax(x)


def resnet18(in_channels, classes_num):
    return ResNet(in_channels, BasicBlock, [2, 2, 2, 2], classes_num)
def resnet34(in_channels, classes_num):
    return ResNet(in_channels, BasicBlock, [3, 4, 6, 3], classes_num)
def resnet50(in_channels, classes_num):
    return ResNet(in_channels, Bottleneck, [3, 4, 6, 3], classes_num)

def main():
    support.set_batch_size(256)
    model = resnet50(10, 4).to(support.DEVICE)
    train_loader,valid_loader = support.get_dataloader()

    support.simple_train(model, train_loader, valid_loader, 100)
    # support.catch_cancel(lambda: support.l2_regex_train(model,0.01,train_loader,valid_loader,100))

    other_dataloader = support.get_dataloader(domain=1, split=False)
    support.test(model, other_dataloader)
    # support.l2_regex_train(model, 0.2,train_loader,valid_loader)

if __name__ == '__main__':
    main()