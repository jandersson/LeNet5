import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from mnist import mnist
from lenet5 import LeNet5
import numpy as np
import argparse
import visdom

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

def update_learning_rate(optimizer, current_epoch, override=None):
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
    if override:
        new_lr = override
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def save_model(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



class Trainer(object):
    def __init__(self):
        self.running_loss = 0.0
        self.epochs = 20
        self.model = None
        self.optimizer = None
        self.vis = visdom.Visdom()
        assert self.vis.check_connection()


        self.test_accuracy_plot = self.vis.line(Y=np.array([0]), X=np.array([0]), opts=dict(
            title='Test Accuracy',
            ylabel='Accuracy',
            xlabel='Epoch'
        ))

    def update_loss_plot(self, epoch, epoch_loss):
        if not self.train_loss_plot:
            self.train_loss_plot = self.vis.line(Y=np.array([epoch_loss]),
                                                 X=np.array([epoch]),
                                                 opts=dict(
                                                    title='Training Loss',
                                                    ylabel='Loss',
                                                    xlabel='Epoch'
                                                ))
        else:
            self.vis.line(Y=np.array([epoch_loss]),
                          X=np.array([epoch]),
                          win=self.train_loss_plot,
                          update='append')

    def update_test_accuracy_plot(self, epoch, accuracy):
        self.vis.line(Y=np.array([accuracy]),
                      X=np.array([epoch]),
                      win=self.test_accuracy_plot,
                      update='append')

    def setup_model(self, resume=False):
        print("Loading Model")
        self.model = LeNet5()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        if resume:
            print("Resuming from saved model")
            self.load_saved_model()
        if torch.cuda.is_available():
            print("Using GPU")
            self.model.cuda()

    def load_saved_model(self, checkpoint='checkpoint.pth.tar'):
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self):
        print("Training Module Started")
        args = get_args()
        self.setup_model()
        resume = args.resume
        start_epoch = 0

        running_loss = 0.0
        print("Loading MNIST Data")
        training_data = DataLoader(mnist(set_type='train'), batch_size=1)
        print("MNIST Loaded")
        print("Transforming Data")
        train_mean = training_data.dataset.pix_mean
        train_stdev = training_data.dataset.stdev
        trsfrms = transforms.Compose([ZeroPad(pad_size=2),
                                      Normalize(mean=train_mean, stdev=train_stdev),
                                      ToTensor()])
        training_data.dataset.transform = trsfrms
        test_data = DataLoader(mnist(set_type='test', transform=trsfrms), batch_size=1)
        print("Transform Complete")
        loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
        # TODO: Plot an error vs training set size curve
        start_time = time.time()
        for t in range(start_epoch, self.epochs):
            update_learning_rate(self.optimizer, t)
            epoch_start_time = time.time()
            epoch_loss = 0
            self.model.train(True)
            for sample in training_data:
                image = Variable(sample['image'])
                # TODO: Detect loss type and do the right transformation on label
                # Do this for MSELoss
                # label = Variable((sample['label'].squeeze() == 1).nonzero(), requires_grad=False)
                # label style for Cross Entropy Loss
                label = Variable(sample['label'].squeeze().nonzero().select(0,0), requires_grad=False)
                y_pred = self.model(image)
                loss = loss_fn(y_pred, label)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.update_loss_plot(t + 1, epoch_loss)
            self.model.train(False)
            correct = 0
            for sample in test_data:
                image = Variable(sample['image'])
                label = Variable(sample['label'])
                y_pred = self.model(image)
                correct += 1 if torch.equal(torch.max(y_pred.data, 1)[1], torch.max(label.data, 1)[1]) else 0
            test_accuracy = correct/len(test_data)
            self.update_test_accuracy_plot(t + 1, test_accuracy)
            print(f"Epoch: {t}\tRunning Loss: {running_loss:.2f}\tEpoch time: {(time.time() - epoch_start_time):.2f} sec")
            print(f"Test Accuracy: {test_accuracy:.2%}")
            print(f"Elapsed time: {(time.time() - start_time):.2f} sec")
            print("Creating checkpoint")
            save_model({'epoch': t, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()})



def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint file')
    return parser.parse_args()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
