import torch
import torch.nn as nn

def conv1x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1)

def ConvBlock(in_channels, out_channels):
    return nn.Sequential(
        conv1x3(in_channels, in_channels),
        nn.ReLU(),
        conv1x3(in_channels, in_channels),
        nn.ReLU(),
        conv1x3(in_channels, out_channels, 2),
        nn.ReLU()
    )

class CNND(nn.Module):
    def __init__(self):
        super(CNND, self).__init__()
        self.conv1 = conv1x3(1, 20)
        self.relu = nn.ReLU()
        self.block1 =ConvBlock(20, 30)
        self.block2 = ConvBlock(30, 40)
        self.block3 = ConvBlock(40, 30)
        self.block4 = ConvBlock(30, 20)
        self.block5 = ConvBlock(20, 20)
        self.pool = nn.Conv1d(20, 1, 1)
        self.fc = nn.Linear(6, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.sigmod(x)
        x = x.flatten()

        return x

if __name__ == "__main__":
    # Hyper-parameters
    learning_rate = 0.001

    model = CNND()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    input = torch.rand(124, 1, 189)
    output = model(input)

    print(input.shape)
    print(output.shape)
