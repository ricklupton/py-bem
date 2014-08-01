import unittest
import numpy as np
from numpy import pi
from numpy.testing import assert_array_equal
from bem import Blade
from io import StringIO

# NB twist is in degrees
yaml = """
x:     [0.2, 3.2, 4.3]
chord: [2.3, 1.0, 3]
twist: [2, 5, 0.2]
thickness: [100, 100, 43]
"""


class TestBlade(unittest.TestCase):
    def test_it_works(self):
        blade = Blade([1], [2], [3], [4])
        self.assertEqual(blade.x, [1])
        self.assertEqual(blade.chord, [2])
        self.assertEqual(blade.twist, [3])
        self.assertEqual(blade.thickness, [4])

    def test_yaml_loading(self):
        blade = Blade.from_yaml(StringIO(yaml))
        assert_array_equal(blade.x, [0.2, 3.2, 4.3])
        assert_array_equal(blade.chord, [2.3, 1.0, 3])
        assert_array_equal(blade.twist, [x*pi/180 for x in [2, 5, 0.2]])
        assert_array_equal(blade.thickness, [100, 100, 43])

    def test_resampling(self):
        blade = Blade.from_yaml(StringIO(yaml))
        x = [1.2, 3.0, 4.3]
        b2 = blade.resample(x)

        assert_array_equal(b2.x, x)
        assert_array_equal(b2.chord, np.interp(x, blade.x, blade.chord))
        assert_array_equal(b2.twist, np.interp(x, blade.x, blade.twist))
        assert_array_equal(b2.thickness,
                           np.interp(x, blade.x, blade.thickness))


if __name__ == '__main__':
    unittest.main()
