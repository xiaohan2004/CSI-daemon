import torch
import torch.nn as nn
import torch.nn.functional as F

import models.support


class Transformer(nn.Module):
    def __init__(self, in_features,num_classes=4, num_heads=4, num_layers=6, len_w=models.support.LEN_W, dropout=0.1):
        super(Transformer, self).__init__()

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(in_features*len_w, num_classes)

    def forward(self, src):
        # Encoder
        output = self.transformer_encoder(src)

        output = self.flatten(output)

        output = self.fc_out(output)
        return output

def main():
    train_loader, test_loader = models.support.get_dataloader(domain=0)
    model = Transformer(in_features=52, num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1000):
        total_loss = 0
        for inputs,labels in train_loader:
            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}, loss {total_loss}")

        with torch.no_grad():
            total = 0
            correct = 0
            for inputs,labels in test_loader:
                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Accuracy: {correct / total}")

if __name__ == '__main__':
    main()