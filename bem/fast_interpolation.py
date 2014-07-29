"""
From http://stackoverflow.com/a/13504757
"""

from scipy.interpolate import interp1d
from scipy.interpolate._fitpack import _bspleval
import numpy as np


class fast_interpolation:
    def __init__(self, x, y, axis=-1):
        assert len(x) == y.shape[axis]
        self.x = x
        self.y = y
        self.axis = axis
        self._f = interp1d(x, y, axis=axis, kind='slinear', copy=False)

    def __getstate__(self):
        return dict(x=self.x, y=self.y, axis=self.axis)

    def __setstate__(self, state):
        self.x = state['x']
        self.y = state['y']
        self.axis = state['axis']
        self._f = interp1d(self.x, self.y, axis=self.axis,
                           kind='slinear', copy=False)

    def __call__(self, new_x):
        #assert new_x.shape == y.shape
        xj, cvals, k = self._f._spline
        result = np.empty_like(new_x)
        for i, value in enumerate(new_x.flat):
            result.flat[i] = _bspleval(value, self.x, cvals[:, i], k, 0)
        return result
