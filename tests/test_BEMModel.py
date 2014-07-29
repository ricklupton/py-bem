import unittest
from numpy import pi, array
from numpy.testing import assert_array_almost_equal as assert_aae

from bem import BEMModel


class BEMModelTestCase(unittest.TestCase):
    root_length = 10

    def setUp(self):
        class MockBlade:
            radii = array([0, 2, 4, 6])
            chord = array([1, 1, 1, 1])
            twist = array([1, 1, 1, 1]) * pi / 180
            thickness = array([1, 1, 1, 1])

        class MockDatabase:
            def for_thickness(self, thickness):
                return array([[-2*pi*pi, 0.2],
                              [+2*pi*pi, 0.2]])
            alpha = array([-pi, pi])

        self.model = BEMModel(MockBlade(), self.root_length,
                              3, MockDatabase())

    def test_annuli_edges_are_correct(self):
        self.assertEqual(list(self.model.boundaries),
                         [10, 11, 13, 15, 16])

    def test_solve_wake_is_same_as_solve(self):
        args = (12.2, 12*pi/30, 0)
        factors = self.model.solve(*args)
        wake = self.model.solve_wake(*args)
        assert_aae(factors[:, 0], wake[:, 0] / args[0])
        assert_aae(factors[:, 1], wake[:, 1] / args[1] / self.model.radii)

    def test_solve_wake_is_same_as_solve_with_extra_factors(self):
        args = (12.2, 12*pi/30, 0)
        wake = self.model.solve_wake(*args)
        extra_velocities = wake * 0.43
        extra_velocity_factors = extra_velocities / args[0]

        factors = self.model.solve(
            *args, extra_velocity_factors=extra_velocity_factors)
        wake = self.model.solve_wake(*args, extra_velocities=extra_velocities)

        assert_aae(factors[:, 0], wake[:, 0] / args[0])
        assert_aae(factors[:, 1], wake[:, 1] / args[1] / self.model.radii)

    def test_lift_drag_one_annulus(self):
        alpha = array([5, 4, 3, 2]) * pi / 180
        all_lift_drag = self.model.lift_drag(alpha)
        one_lift_drag = self.model.lift_drag(alpha[2:3], annuli=slice(2, 3))
        assert_aae(one_lift_drag, all_lift_drag[2:3, :])

    def test_force_coefficients_one_annulus(self):
        phi = array([5, 4, 3, 2]) * pi / 180
        all_coeffs = self.model.force_coefficients(phi, 0)
        one_coeffs = self.model.force_coefficients(phi[2:3], 0, slice(2, 3))
        assert_aae(one_coeffs, all_coeffs[2:3, :])

    def test_inflow_derivatives_with_one_annulus(self):
        args = (12.2, 12*pi/30, 0)
        factors = self.model.solve(*args)

        all_derivs = self.model.inflow_derivatives(*args, factors=factors*1.1)

        one_derivs = self.model.inflow_derivatives(*args,
                                                   factors=factors[2:3]*1.1,
                                                   annuli=slice(2, 3))

        assert_aae(one_derivs, all_derivs[2:3, :])

    def test_one_annulus_fails_with_wrong_shapes(self):
        args = (12.2, 12*pi/30, 0)
        factors = self.model.solve(*args)

        with self.assertRaises(Exception):
            self.model.inflow_derivatives(*args,
                                          factors=factors[2:4],
                                          annuli=slice(2, 3))

        with self.assertRaises(Exception):
            phi = array([5, 4, 3, 2]) * pi / 180
            self.model.force_coefficients(phi[2:4], 0, slice(2, 3))

        with self.assertRaises(Exception):
            alpha = array([5, 4, 3, 2]) * pi / 180
            self.model.lift_drag(alpha[2:4], annuli=slice(2, 3))
