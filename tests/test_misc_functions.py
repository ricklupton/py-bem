import unittest
from numpy import pi, sin, cos, array
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae

from bem import iterate_induction_factors, inflow
from bem.bem import _strip_boundaries, _thrust_correction_factor


class strip_boundaries_TestCase(unittest.TestCase):
    def test_boundaries(self):
        #         0   1   2   3   4   5  6   7
        # Input:  X   X   .   X   .   .  X   X
        # Result: | |     |         |      | |
        radii = [0, 1, 3, 6, 7]

        expected = [0.0, 0.5, 2.0, 4.5, 6.5, 7.0]
        result = _strip_boundaries(radii)
        self.assertEqual(list(result), expected)


class thrust_correction_factor_TestCase(unittest.TestCase):
    def test_no_correction_for_small_a(self):
        self.assertEqual(_thrust_correction_factor(0), 1)

    def test_smooth_transition(self):
        a_transition = 0.3539
        H1 = _thrust_correction_factor(a_transition)
        H2 = _thrust_correction_factor(a_transition + 1e-4)
        self.assertLess((H1 - H2), 1e-3)


class iterate_induction_factors_TestCase(unittest.TestCase):
    def test_there_is_no_induction_if_no_lift(self):
        force_coeffs = np.array([[0, 0]])
        factors = iterate_induction_factors(1, force_coeffs, 0.21, 0,
                                            array([[0, 0]]))
        self.assertEqual(factors[0, 0], 0)
        self.assertEqual(factors[0, 0], 0)

    def test_imitating_lift_coefficient(self):
        # Try to set lift coefficient so that thrust force is equal to
        # expected value from momentum theory. This should mean the
        # calculation is already converged.

        # Choose some values
        U = 5.4     # wind speed
        w = 1.2     # rotor speed
        r = 15.3    # radius
        a = 0.14    # induction factor
        Nb = 3      # number of blades
        c = 1.9     # chord

        def lift_drag(alpha):
            # Thrust should be 2 rho A U^2 a (1-a)`
            #   for annulus, A = 2 pi r  dr
            # If no drag, then
            #   thrust on blade = 0.5 rho c (U(1-a)/sin(phi))^2 CL dr cos(phi)
            # So CL = (8 pi r a sin^2 phi) / ((1-a) Nb c cos(phi))
            CL = 8*pi*r * a * sin(alpha)**2 / ((1-a) * Nb * c * cos(alpha))
            return [CL[0], 0]

        LSR = w * r / U
        solidity = Nb * c / (2*pi*r)
        factors = np.array([[a, 0]])

        Wnorm, phi = inflow(LSR, factors)
        cphi, sphi = np.cos(phi[0]), np.sin(phi[0])
        A = array([[cphi, sphi], [-sphi, cphi]])
        force_coeffs = np.array([np.dot(A, lift_drag(phi))])
        factors = iterate_induction_factors(LSR, force_coeffs,
                                            solidity, 0, factors)
        assert_aae(a, factors[0, 0])
        assert np.all(factors[:, 1] > 0)
        # XXX could check at better


class inflow_TestCase(unittest.TestCase):
    def assertInflow(self, lsr, factors, extra, expected):
        if extra is not None:
            extra = np.atleast_2d(extra)
        factors = np.atleast_2d(factors)
        self.assertEqual(inflow(lsr, factors, extra), expected)

    def test_simple_cases(self):
        # Zero LSR -> not rotating.
        self.assertInflow(0, (0, 0), None, (1, pi/2))

        # Zero LSR, axial induction -> no flow or double flow
        self.assertInflow(0, (1, 0), None, (0, 0))
        self.assertInflow(0, (-1, 0), None, (2, pi/2))

        # LSR = 1 -> angle should be 45 deg with no induction
        self.assertInflow(1, (0, 0), None, (2**0.5, pi/4))

        # LSR = 1, axial induction of 1 -> flow should be in-plane
        self.assertInflow(1, (1, 0), None, (1, 0))

        # LSR = 1, axial induction of 1, tangential inflow of +/-1
        #  -> flow should be in-plane, zero or double
        self.assertInflow(1, (1, -1), None, (0, 0))
        self.assertInflow(1, (1, 1), None, (2, 0))

    def test_blade_velocities(self):
        # Zero LSR, blade moving downwind at windspeed -> no flow
        self.assertInflow(0, (0, 0), (1, 0), (0, 0))

        # Zero LSR, blade moving upwind at windspeed -> double flow
        self.assertInflow(0, (0, 0), (-1, 0), (2, pi/2))
