import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):

    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))


        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))

    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        pass

class CNN_small(nn.Module):
    """
    Small CNN for CIFAR-10 warm-up.

    Hidden (parameterized) layers:
      1) conv1
      2) conv2
      3) fc1
    Output layer:
      4) fc2

    BatchNorm / MaxPool / Dropout do NOT count toward the hidden-layer limit.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Convolutional part
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two 2x2 pools:
        # input: 3 x 32 x 32
        # conv1 -> 32 x 32 x 32
        # pool  -> 32 x 16 x 16
        # conv2 -> 64 x 16 x 16
        # pool  -> 64 x 8 x 8
        flattened_dim = 64 * 8 * 8

        # Fully connected part
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # raw logits

        return x
