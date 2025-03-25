import torch

import GRU
import support

def main():
    loss_min = None
    model = GRU.GRU(hidden_dim=32, num_classes=4).to(support.DEVICE)
    train_loader, test_loader = support.get_dataloader()
    def save(loss):
        global loss_min
        if loss_min is None or loss_min > loss:
            loss_min = loss
            torch.save(model.state_dict(), '../saved/denoise_model.pth')
    support.l2_regex_train(model, 0.2, train_loader, test_loader, 1000, save)


if __name__ == '__main__':
    main()