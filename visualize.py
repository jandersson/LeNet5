"""Visualization and Logging with Visdom"""
import visdom


class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom()
        assert self.vis.check_connection()
