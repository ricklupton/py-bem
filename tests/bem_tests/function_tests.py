from nose.tools import *
import unittest

from numpy import pi, sin, cos, array
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae

from bem.bem import (Aerofoil, BladeSection,
                     thrust_correction_factor, iterate_induction_factors,
                     inflow, strip_boundaries)


class strip_boundaries_TestCase(unittest.TestCase):
    def test_boundaries(self):
        #         0   1   2   3   4   5  6   7
        # Input:  X   X   .   X   .   .  X   X
        # Result: | |     |         |      | |
        radii = [0, 1, 3, 6, 7]

        expected = [0.0, 0.5, 2.0, 4.5, 6.5, 7.0]
        result = strip_boundaries(radii)
        self.assertEqual(list(result), expected)


class thrust_correction_factor_test:
    def test_no_correction_for_small_a(self):
        eq_(thrust_correction_factor(0), 1)

    def test_smooth_transition(self):
        a_transition = 0.3539
        H1 = thrust_correction_factor(a_transition)
        H2 = thrust_correction_factor(a_transition + 1e-4)
        assert (H1 - H2) < 1e-3


class iterate_induction_factors_test:
    def test_there_is_no_induction_if_no_lift(self):
        foil = Aerofoil('test', lambda alpha: [0, 0])
        force_coeffs = np.array([[0, 0]])
        factors = iterate_induction_factors(1, force_coeffs, 0.21, 0,
                                          array([[0, 0]]))
        eq_(factors[0, 0], 0)
        eq_(factors[0, 0], 0)

    def test_imitating_lift_coefficient(self):
        # Try to set lift coefficient so that thrust force is equal to
        # expected value from momentum theory. This should mean the
        # calculation is already converged.

        # Choose some values
        U  = 5.4    # wind speed
        w  = 1.2    # rotor speed
        r  = 15.3   # radius
        a  = 0.14   # induction factor
        Nb = 3      # number of blades
        c  = 1.9    # chord

        def lift_drag(alpha):
            # Thrust should be 2 rho A U^2 a (1-a)`
            #   for annulus, A = 2 pi r  dr
            # If no drag, then
            #   thrust on blade = 0.5 rho c (U(1-a)/sin(phi))^2 CL dr cos(phi)
            # So CL = (8 pi r a sin^2 phi) / ((1-a) Nb c cos(phi))
            CL = 8*pi*r * a * sin(alpha)**2 / ((1-a) * Nb * c * cos(alpha))
            return [CL, 0]

        LSR = w * r / U
        solidity = Nb * c / (2*pi*r)
        factors = np.array([[a, 0]])
        force_coeffs = np.array([lift_drag(inflow(LSR, factors)[1])])
        factors = iterate_induction_factors(LSR, force_coeffs,
                                            solidity, 0, factors)
        assert_aae(a, factors[0, 0])
        assert at1 > 0
        # XXX could check at better

# class solve_induction_factors_test:
#     def test_it_converges_with_reasonable_input_values(self):
#         # Choose some values
#         U  = 5.4    # wind speed
#         w  = 1.2    # rotor speed
#         r  = 15.3   # radius
#         Nb = 3      # number of blades
#         c  = 1.9    # chord

#         lift_drag = lambda a: array([2*pi, 0.10]) * a
#         foil = Aerofoil('test', lift_drag)

#         LSR = w * r / U
#         solidity = Nb * c / (2*pi*r)
#         section = BladeSection(chord=c, twist=0, foil=foil)
#         a, at = solve_induction_factors(LSR, section, solidity, pitch=0)
#         assert a  > 0 and a  < 0.4
#         assert at > 0 and at < 0.2

#     def test_it_gives_up_with_unreasonable_input_values(self):
#         # Choose some values
#         U  = 5.4    # wind speed
#         w  = 1.2    # rotor speed
#         r  = 15.3   # radius
#         Nb = 3      # number of blades
#         c  = 1.9    # chord

#         lift_drag = lambda a: array([np.random.random()*2 - 1, 0.10]) * a
#         foil = Aerofoil('test', lift_drag)

#         LSR = w * r / U
#         solidity = Nb * c / (2*pi*r)
#         section = BladeSection(chord=c, twist=0, foil=foil)
#         with assert_raises(RuntimeError):
#             a, at = solve_induction_factors(LSR, section, solidity, 0)


class inflow_test:
    def test_simple_cases(self):
        # Zero LSR -> not rotating.
        assert_equal(inflow(0, (0, 0)), (1, pi/2))

        # Zero LSR, axial induction -> no flow or double flow
        assert_equal(inflow(0, (1, 0)), (0, 0))
        assert_equal(inflow(0, (-1, 0)), (2, pi/2))

        # LSR = 1 -> angle should be 45 deg with no induction
        assert_equal(inflow(1, (0, 0)), (2**0.5, pi/4))

        # LSR = 1, axial induction of 1 -> flow should be in-plane
        assert_equal(inflow(1, (1, 0)), (1, 0))

        # LSR = 1, axial induction of 1, tangential inflow of +/-1
        #  -> flow should be in-plane, zero or double
        assert_equal(inflow(1, (1, -1)), (0, 0))
        assert_equal(inflow(1, (1, 1)), (2, 0))

    def test_blade_velocities(self):
        # Zero LSR, blade moving downwind at windspeed -> no flow
        assert_equal(inflow(0, (0, 0), (1, 0)), (0, 0))

        # Zero LSR, blade moving upwind at windspeed -> double flow
        assert_equal(inflow(0, (0, 0), (-1, 0)), (2, pi/2))
