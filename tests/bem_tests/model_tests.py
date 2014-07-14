from nose.tools import *

from numpy import pi, sin, cos, array
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae

from bem.bem import (AerofoilDatabase, Aerofoil, BladeSection, BEMModel,
                     thrust_correction_factor, iterate_induction_factors)


class BEMModel_Tests:
    root_length = 10

    def setup(self):
        class MockBlade:
            radii = array([0, 2, 4, 6])
            chord = array([1, 1, 1, 1])
            twist = array([1, 1, 1, 1])
            thickness = array([1, 1, 1, 1])
        class MockDatabase:
            def for_thickness(self, thickness):
                return array([[0, 0], [0, 0]])
            alpha = [-pi, pi]
        self.model = BEMModel(MockBlade(), self.root_length,
                              3, MockDatabase(), unsteady=True)

    def test_annuli_edges_are_correct(self):
        eq_(list(self.model.boundaries), [10, 11, 13, 15, 16])

    def test_solve_wake_is_same_as_solve(self):
        args = (12.2, 12*pi/30, 0)
        factors = self.model.solve(*args)
        wake = self.model.solve_wake(*args)
        assert_aae(factors[:, 0], wake[:, 0] / args[0])
        assert_aae(factors[:, 1], wake[:, 1] / args[1] / self.model.radii)
