"""Visualization and Logging with Visdom"""
import visdom
import numpy as np


class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom()
        assert self.vis.check_connection()

        # Visdom requires one data point for initializing a plot. No blankboards allowed yet.
        self.train_loss_plot = None
        self.test_accuracy_plot = None
        self.text_log = None
        self.log_messages = ''

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
        if not self.test_accuracy_plot:
            self.test_accuracy_plot = self.vis.line(Y=np.array([accuracy]),
                                                    X=np.array([epoch]),
                                                    opts=dict(
                                                        title='Test Accuracy',
                                                        ylabel='Accuracy',
                                                        xlabel='Epoch'
                                                    ))
        else:
            self.vis.line(Y=np.array([accuracy]),
                          X=np.array([epoch]),
                          win=self.test_accuracy_plot,
                          update='append')

    def write_log(self, message):
        print(message)
        if not self.text_log:
            self.log_messages = message
            self.text_log = self.vis.text(message)
        else:
            self.log_messages = self.log_messages + f"\n<br>{message}"
            self.vis.text(self.log_messages, win=self.text_log)

