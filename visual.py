from visdom import Visdom
import numpy as np
Visdom_default_colors = {
    0: [180, 119, 31],    # Blue → BGR
    1: [14, 127, 255],    # Orange → BGR
    2: [44, 160, 44],     # Green
    3: [40, 39, 214],     # Red
    4: [189, 103, 148],   # Purple
    5: [75, 86, 140],     # Brown
    6: [194, 119, 227],   # Pink
    7: [127, 127, 127],   # Gray
    8: [34, 189, 188],    # Olive
    9: [207, 190, 23],    # Cyan
}
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8091)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name ,
                xlabel='Epochs',
                ylabel= title_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')