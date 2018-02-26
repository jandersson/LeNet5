"""PyTorch Implementation of LeNet-5, Recreating the network in http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf """

import pathlib
import gzip
import struct
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset

class mnist(Dataset):
    def __init__(self):
        """Data and data format specification at http://yann.lecun.com/exdb/mnist/"""
        self.train = []
        self.test = []
        train_images, train_labels, test_images, test_labels = [[] for _ in range(4)]
        data_path = pathlib.Path(__file__).parents[1] / 'data' / 'mnist'
        for compressed_file in data_path.glob('*.gz'):
            with gzip.open(compressed_file, 'rb') as cf:
                if 'labels' in compressed_file.name:
                    # Unpack magic number (discarded) and number of elements
                    _magic_number, num_labels = struct.unpack('>ii', cf.read(8))
                    # Unpack the rest into a list
                    labels_iter = struct.iter_unpack('>B', cf.read())
                    labels = [label[0] for label in labels_iter]
                    if 't10k' in compressed_file.name:
                        test_labels = labels
                    elif 'train' in compressed_file.name:
                        train_labels = labels
                    assert num_labels == len(labels)
                elif 'images' in compressed_file.name:
                    # Unpack the magic number (discarded), number of images, number of rows, number of columns
                    images = []
                    _magic_number, num_images, self.num_rows, self.num_cols = struct.unpack('>iiii', cf.read(16))
                    pixels = list(struct.iter_unpack('>B', cf.read()))
                    for i in range(0, num_images * self.num_rows * self.num_cols, self.num_rows * self.num_cols):
                        images.append([pixel[0] for pixel in pixels[i: i + self.num_rows * self.num_cols]])
                    if 't10k' in compressed_file.name:
                        test_images = images
                    elif 'train' in compressed_file.name:
                        train_images = images
                    assert len(images) == num_images
        assert len(train_images) == len(train_labels)
        assert len(test_images) == len(test_labels)
        self.train = list(zip(train_images, train_labels))
        self.test = list(zip(test_images, test_labels))

    def __len__(self):
        return len(self.train) + len(self.test)

class LeNet5(torch.nn.Module):
    """LeNet-5 CNN Architecture"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Tanh(),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Tanh(),
        )
        self.c3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=120, stride=1, kernel_size=5, padding=0),
            torch.nn.Tanh(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=10),
        )
    def forward(self, x):
        c1_out = self.c1(x)
        c2_out = self.c2(c1_out)
        c3_out = self.c3(c2_out)
        c3_flat = c3_out.view(c3_out.size(0), -1)
        return self.classifier(c3_flat)

if __name__ == '__main__':
    data = mnist()
    model = LeNet5()
    input_pad_width = 2
    loss_fn = torch.nn.MSELoss(size_average=True)
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # TODO: Use same learning rate in paper
    # TODO: Use same optimization strategy in paper
    # TODO: Run more than one epoch
    # TODO: Check test error
    # TODO: Plot an error vs training set size curve
    # TODO: Plot an epoch vs error curve
    # TODO: Implement argparse
    # TODO: Normalize image data to [-0.1, 1.175]
    # TODO: Track training time
    for t in range(1):
        # TODO: Incomplete
        error = []
        for image, label in data.train:
            image = np.array(image).reshape([data.num_rows, data.num_cols])
            image = np.pad(image, input_pad_width, 'edge')
            image = image.reshape([1, 1, image.shape[0], image.shape[1]])
            label_vec = [1 if lbl == label else 0 for lbl in range(10)]
            x = Variable(torch.FloatTensor(image).type(dtype))
            label = Variable(torch.FloatTensor(label_vec), requires_grad=False).type(dtype)
            y_pred = model(x)
            loss = loss_fn(y_pred, label)
            print(t, loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(t)

