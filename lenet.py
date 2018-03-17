"""PyTorch Implementation of LeNet-5, Recreating the network in http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf """

import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import mnist


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        dtype = torch.FloatTensor
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = np.array([1 if lbl == label else 0 for lbl in range(10)])
        image = torch.from_numpy(image).type(dtype)
        label = torch.from_numpy(label).type(dtype)
        return {'image': image,
                'label': label}


class ZeroPad(object):
    def __init__(self, pad_size):
        self.pad_size = [(pad_size, pad_size), (pad_size, pad_size), (0, 0)]
    def __call__(self, sample):
        sample['image'] = np.pad(sample['image'], self.pad_size, mode='constant')
        return sample


class Normalize(object):
    """Make the mean input 0 and variance roughly 1 to accelerate learning"""
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def __call__(self, sample):
        original_shape = sample['image'].shape
        image = sample['image'].ravel()
        image -= self.mean
        image /= self.stdev
        image.shape = original_shape
        sample['image'] = image
        return sample


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

def get_optimizer(model, current_epoch):
    """Return optimizer with learning rate schedule from paper"""
    # Learning Rate schedule: 0.0005 for first 2 iterations, 0.0002 for next 3, 0.0001 next 3, 0.00005 next 4,
    # 0.00001 thereafter
    if current_epoch < 2:
        new_lr = 5e-4
    elif current_epoch < 5:
        new_lr = 2e-4
    elif current_epoch < 8:
        new_lr = 1e-4
    elif current_epoch < 12:
        new_lr = 5e-5
    else:
        new_lr = 1e-5
    print(f"Using learning rate {new_lr} for epoch {current_epoch}")
    return torch.optim.SGD(model.parameters(), lr=5e-4)


if __name__ == '__main__':
    training_data = DataLoader(mnist(set_type='train'), batch_size=1)
    train_mean = training_data.dataset.pix_mean
    train_stdev = training_data.dataset.stdev
    trsfrms = transforms.Compose([ZeroPad(pad_size=2),
                                  Normalize(mean=train_mean, stdev=train_stdev),
                                  ToTensor()])
    training_data.dataset.transform = trsfrms
    test_data = DataLoader(mnist(set_type='test', transform=trsfrms), batch_size=1)
    model = LeNet5()
    loss_fn = torch.nn.MSELoss(size_average=True)
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        print("Using GPU")
        dtype = torch.cuda.FloatTensor
        model.cuda()
    # TODO: Plot an error vs training set size curve
    # TODO: Plot an epoch vs error curve
    # TODO: Implement argparse
    EPOCHS = 40
    running_loss = 0.0
    start_time = time.time()
    for t in range(EPOCHS):
        # TODO: Incomplete
        error = []
        optimizer = get_optimizer(model, t)
        epoch_start_time = time.time()
        model.train(True)
        for sample in training_data:
            image = Variable(sample['image'])
            label = Variable(sample['label'], requires_grad=False)
            y_pred = model(image)
            loss = loss_fn(y_pred, label)
            running_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.train(False)
        correct = 0
        for sample in test_data:
            image = Variable(sample['image'])
            label = Variable(sample['label'])
            y_pred = model(image)
            correct += 1 if torch.equal(torch.max(y_pred.data, 1)[1], torch.max(label.data, 1)[1]) else 0
        print(f"Epoch: {t}\tRunning Loss: {running_loss:.2f}\tEpoch time: {(time.time() - epoch_start_time):.2f} sec")
        print(f"Test Accuracy: {(correct/len(test_data)):.2%}")
        print(f"Elapsed time: {(time.time() - start_time):.2f} sec")

