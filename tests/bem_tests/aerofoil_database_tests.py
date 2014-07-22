from nose.tools import *

from numpy import pi
from scipy.interpolate import interp1d
from numpy.testing import assert_array_almost_equal as assert_aae

from bem.bem import AerofoilDatabase


class AerofoilDatabase_Tests:
    def setup(self):
        self.db = AerofoilDatabase('tests/data/aerofoils.npz')

    # CylinderData:
    def test_cylinder_data_read_correctly(self):
        lift_drag = self.db.for_thickness(1.0)
        assert_aae(lift_drag[:, 0], 0)  # lift
        assert_aae(lift_drag[:, 1], 1)  # drag

    # ThirteenPercentFoil:
    def test_thirteen_percent_foil_read_correctly(self):
        lift_drag_values = self.db.for_thickness(0.13)
        lift_drag = interp1d(self.db.alpha, lift_drag_values, axis=0)
        assert_aae(lift_drag(0), [0.420, 0.006])
        assert_aae(lift_drag(10*pi/180), [1.460, 0.016])

    def test_thirteen_percent_foil_interpolated_by_alpha(self):
        # Interpolate between 4 and 6 deg alpha
        lift_drag_values = self.db.for_thickness(0.13)
        lift_drag = interp1d(self.db.alpha, lift_drag_values, axis=0)
        assert_aae(lift_drag(5*pi/180), [(0.890 + 1.100) / 2,
                                         (0.009 + 0.012) / 2])

    # InterpolatedData:
    def interpolated_by_thickness(self):
        # Interpolate between 13% and 17% data
        lift_drag_values = self.db.for_thickness(0.15)
        lift_drag = interp1d(self.db.alpha, lift_drag_values, axis=0)
        assert_aae(lift_drag(10*pi/180), [(1.460 + 1.500) / 2,
                                          (0.016 + 0.014) / 2])
