from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from matplotlib.colors import Normalize


def model_accuracy(model, X, Y):
    predictions = model.predict(X)
    rms = sqrt(mean_squared_error(Y, predictions))
    return rms


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

