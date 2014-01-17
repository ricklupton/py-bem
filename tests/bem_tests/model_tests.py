from nose.tools import *

from numpy import pi, sin, cos, array
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae

from bem.bem import (AerofoilDatabase, Aerofoil, BladeSection, BEMModel,
                     thrust_correction_factor, iterate_induction_factors,
                     solve_induction_factors)


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
                return None
        self.model = BEMModel(MockBlade(), self.root_length,
                              3, MockDatabase(), unsteady=True)

    def test_annuli_edges_are_correct(self):
        edges = [annulus.edge_radii for annulus in self.model.annuli]
        eq_(edges, [
            (10, 11),
            (11, 13),
            (13, 15),
            (15, 16),
        ])
