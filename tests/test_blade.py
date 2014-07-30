import unittest
from unittest.mock import sentinel
from numpy import pi
from numpy.testing import assert_array_equal
from bem import Blade
from io import StringIO

# NB twist is in degrees
yaml = """
radii: [0.2, 3.2, 4.3]
chord: [2.3, 1.0, 3]
twist: [2, 5, 0.2]
thickness: [100, 100, 43]
"""


class TestBlade(unittest.TestCase):
    def test_it_works(self):
        blade = Blade([1], [2], [3], [4])
        self.assertEqual(blade.radii, [1])
        self.assertEqual(blade.chord, [2])
        self.assertEqual(blade.twist, [3])
        self.assertEqual(blade.thickness, [4])

    def test_yaml_loading(self):
        blade = Blade.from_yaml(StringIO(yaml))
        assert_array_equal(blade.radii, [0.2, 3.2, 4.3])
        assert_array_equal(blade.chord, [2.3, 1.0, 3])
        assert_array_equal(blade.twist, [x*pi/180 for x in [2, 5, 0.2]])
        assert_array_equal(blade.thickness, [100, 100, 43])


if __name__ == '__main__':
    unittest.main()
