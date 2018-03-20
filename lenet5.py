"""PyTorch Implementation of LeNet-5, Recreating the network in http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
Uses ReLU instead of tanh activation
Uses Max pooling instead of average
"""

import torch

class LeNet5(torch.nn.Module):
    """LeNet-5 CNN Architecture"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.c3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=120, stride=1, kernel_size=5, padding=0),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        c3_out = self.c3(c2_out)
        c3_flat = c3_out.view(c3_out.size(0), -1)
        return self.classifier(c3_flat)
