import time
from datetime import datetime
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from mnist import mnist
from lenet5 import LeNet5
import numpy as np
from visualize import Visualizer

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
    """Deprecated: Return optimizer with learning rate schedule from paper"""
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
        self.current_epoch = 0
        self.epoch_start_time = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.vis = Visualizer()

    def setup_model(self, resume=False):
        print("Loading Model")
        self.model = LeNet5()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                  milestones=[2, 5, 8, 12],
                                                  gamma=0.1)

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

    def load_data(self):
        self.vis.write_log("Loading and Preprocessing MNIST Data")
        self.training_data = DataLoader(mnist(set_type='train'), batch_size=1)
        train_mean = self.training_data.dataset.pix_mean
        train_stdev = self.training_data.dataset.stdev
        trsfrms = transforms.Compose([ZeroPad(pad_size=2),
                                      Normalize(mean=train_mean, stdev=train_stdev),
                                      ToTensor()])
        self.training_data.dataset.transform = trsfrms
        self.test_data = DataLoader(mnist(set_type='test', transform=trsfrms), batch_size=1)
        self.vis.write_log("Loading & Preprocessing Finished")

    def run(self):
        """Run training module, train then test"""
        self.vis.write_log(f"Training Module Started at {datetime.now().isoformat(' ', timespec='seconds')}")
        args = get_args()
        self.setup_model()
        self.loss_fn = torch.nn.CrossEntropyLoss(size_average=True)
        self.load_data()
        resume = args.resume
        self.running_loss = 0.0
        self.start_time = time.time()
        start_epoch = 0
        for self.current_epoch in range(start_epoch, self.epochs):
            self.epoch_start_time = time.time()
            self.train()
            self.test()
            self.vis.write_log("Creating checkpoint")
            save_model({'epoch': self.current_epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()})

    def train(self):
        """Does one training iteration"""
        epoch_loss = 0
        self.model.train(True)
        for sample in self.training_data:
            image = Variable(sample['image'])
            # TODO: Detect loss type and do the right transformation on label
            # Do this for MSELoss
            # label = Variable((sample['label'].squeeze() == 1).nonzero(), requires_grad=False)
            # label style for Cross Entropy Loss
            label = Variable(sample['label'].squeeze().nonzero().select(0,0), requires_grad=False)
            y_pred = self.model(image)
            loss = self.loss_fn(y_pred, label)
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        self.running_loss += epoch_loss
        self.vis.update_loss_plot(self.current_epoch + 1, epoch_loss)

    def test(self):
        """Tests model using test set"""
        self.model.train(False)
        correct = 0
        for sample in self.test_data:
            image = Variable(sample['image'])
            label = Variable(sample['label'])
            y_pred = self.model(image)
            correct += 1 if torch.equal(torch.max(y_pred.data, 1)[1], torch.max(label.data, 1)[1]) else 0
        test_accuracy = correct/len(self.test_data)
        self.vis.update_test_accuracy_plot(self.current_epoch + 1, test_accuracy)
        self.vis.write_log(f"Epoch: {self.current_epoch + 1}\tRunning Loss: {self.running_loss:.2f}\tEpoch time: {(time.time() - self.epoch_start_time):.2f} sec")
        self.vis.write_log(f"Test Accuracy: {test_accuracy:.2%}")
        self.vis.write_log(f"Elapsed time: {(time.time() - self.start_time):.2f} sec")

def get_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint file')
    return parser.parse_args()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
