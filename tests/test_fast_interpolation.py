"""
From http://stackoverflow.com/a/13504757
"""

import unittest
from scipy.interpolate import interp1d
import numpy as np
from bem.fast_interpolation import fast_interpolation
import pickle


# Simple interpolation along middle axis, at each point within y
def original_interpolation(new_x, x, y):
    result = np.empty((y.shape[0], y.shape[2]))
    for i in range(y.shape[0]):
        for j in range(y.shape[2]):
            f = interp1d(x, y[i, :, j], axis=-1, kind='slinear')
            result[i, j] = f(new_x[i, j])
    return result


class FastInterpolationTestCase(unittest.TestCase):
    def test_interpolation(self):
        # Interpolate along y
        nx, ny, nz = 30, 40, 2
        x = np.arange(0, ny, 1.0)
        y = np.random.randn(nx, ny, nz)
        new_x = np.random.random_integers(1, (ny-1)*10, size=(nx, nz))/10.0

        r1 = original_interpolation(new_x, x, y)
        r2 = fast_interpolation(x, y, axis=1)
        np.testing.assert_allclose(r1, r2(new_x))

    def test_picklable(self):
        # Interpolate along y
        nx, ny, nz = 30, 40, 2
        x = np.arange(0, ny, 1.0)
        y = np.random.randn(nx, ny, nz)
        new_x = np.random.random_integers(1, (ny-1)*10, size=(nx, nz))/10.0

        orig = fast_interpolation(x, y, axis=1)
        pickled = pickle.loads(pickle.dumps(orig))
        np.testing.assert_allclose(orig(new_x), pickled(new_x))


if __name__ == '__main__':
    unittest.main()
