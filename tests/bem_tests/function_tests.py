from nose.tools import *

from numpy import pi, sin, cos, array
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae

from bem.bem import (AerofoilDatabase, Aerofoil, BladeSection, BEMModel,
                     thrust_correction_factor, iterate_induction_factors,
                     solve_induction_factors)


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
        foil = Aerofoil('test', lambda alpha: 0, lambda alpha: 0)
        section = BladeSection(chord=1, twist=0.43, foil=foil)
        a, at = iterate_induction_factors(1, section, 0.21, 0, (0, 0))
        eq_(a, 0)
        eq_(at, 0)

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

        def CL(alpha):
            # Thrust should be 2 rho A U^2 a (1-a)
            #   for annulus, A = 2 pi r  dr
            # If no drag, then
            #   thrust on blade = 0.5 rho c (U(1-a)/sin(phi))^2 CL dr cos(phi)
            # So CL = (8 pi r a sin^2 phi) / ((1-a) Nb c cos(phi))
            return 8*pi*r * a * sin(alpha)**2 / ((1-a) * Nb * c * cos(alpha))
        foil = Aerofoil('test', CL, CD=lambda alpha: 0)

        LSR = w * r / U
        solidity = Nb * c / (2*pi*r)
        section = BladeSection(c, twist=0, foil=foil)
        a1, at1 = iterate_induction_factors(LSR, section, solidity, 0, (a, 0))
        assert_aae(a, a1)
        assert at1 > 0
        # XXX could check at better

class solve_induction_factors_test:
    def test_it_converges_with_reasonable_input_values(self):
        # Choose some values
        U  = 5.4    # wind speed
        w  = 1.2    # rotor speed
        r  = 15.3   # radius
        Nb = 3      # number of blades
        c  = 1.9    # chord

        foil = Aerofoil('test',
                        CL=lambda alpha: 2*pi*alpha,
                        CD=lambda alpha: 0.10*alpha)

        LSR = w * r / U
        solidity = Nb * c / (2*pi*r)
        section = BladeSection(chord=c, twist=0, foil=foil)
        a, at = solve_induction_factors(LSR, section, solidity, pitch=0)
        assert a  > 0 and a  < 0.4
        assert at > 0 and at < 0.2

    def test_it_gives_up_with_unreasonable_input_values(self):
        # Choose some values
        U  = 5.4    # wind speed
        w  = 1.2    # rotor speed
        r  = 15.3   # radius
        Nb = 3      # number of blades
        c  = 1.9    # chord

        foil = Aerofoil('test',
                        CL=lambda alpha: np.random.random()*2 - 1,
                        CD=lambda alpha: 0.10*alpha)

        LSR = w * r / U
        solidity = Nb * c / (2*pi*r)
        section = BladeSection(chord=c, twist=0, foil=foil)
        with assert_raises(RuntimeError):
            a, at = solve_induction_factors(LSR, section, solidity, 0)

