from nose.tools import assert_raises
from mock import Mock
from pyvows import Vows, expect
import __init__  # noqa

from numpy import pi, sin, cos, array
import numpy as np

from bem import (AerofoilDatabase, Aerofoil, BladeSection, BEMModel,
                 thrust_correction_factor, iterate_induction_factors,
                 solve_induction_factors)


@Vows.batch
class AerofoilDatabaseTests(Vows.Context):
    def topic(self):
        return AerofoilDatabase('../aerofoils.npz')

    class CylinderData:
        def topic(self, db):
            return db.for_thickness(1.0)

        def read_correctly(self, foil):
            for alpha in [-0.3, 0, 0.3]:
                expect(foil.CL(alpha)).to_almost_equal(0)
                expect(foil.CD(alpha)).to_almost_equal(1)

    class ThirteenPercentFoil:
        def topic(self, db):
            return db.for_thickness(0.13)

        def read_correctly(self, foil):
            expect(foil.CL(0)).to_almost_equal(0.420)
            expect(foil.CD(0)).to_almost_equal(0.006)
            expect(foil.CL(10*pi/180)).to_almost_equal(1.460)
            expect(foil.CD(10*pi/180)).to_almost_equal(0.016)

        def interpolated_by_alpha(self, foil):
            # Interpolate between 4 and 6 deg alpha
            expect(foil.CL(5*pi/180)).to_almost_equal((0.890 + 1.100) / 2)
            expect(foil.CD(5*pi/180)).to_almost_equal((0.009 + 0.012) / 2)

    class InterpolatedData:
        def topic(self, db):
            # Interpolate between 13% and 17% data
            return db.for_thickness(0.15)

        def interpolated_by_thickness(self, foil):
            expect(foil.CL(10*pi/180)).to_almost_equal((1.460 + 1.500) / 2)
            expect(foil.CD(10*pi/180)).to_almost_equal((0.016 + 0.014) / 2)


@Vows.batch
class BladeSectionTests(Vows.Context):
    def topic(self):
        return BladeSection(1, 2, 3)

    def holds_chord_twist_and_foil(self, section):
        expect(section.chord).to_equal(1)
        expect(section.twist).to_equal(2)
        expect(section.foil).to_equal(3)


@Vows.batch
class BEM_functions(Vows.Context):

    class thrust_correction_factor_test:
        def test_no_correction_for_small_a(self, topic):
            expect(thrust_correction_factor(0)).to_equal(1)

        def test_smooth_transition(self, topic):
            a_transition = 0.3539
            H1 = thrust_correction_factor(a_transition)
            H2 = thrust_correction_factor(a_transition + 1e-4)
            assert (H1 - H2) < 1e-3

    class iterate_induction_factors_test:

        class AnAerofoilWithNoLiftOrDrag:
            def topic(self):
                foil = Aerofoil('test', lambda alpha: 0, lambda alpha: 0)
                return BladeSection(chord=1, twist=0.43, foil=foil)

            def there_is_no_induction(self, section):
                a, at = iterate_induction_factors(1, section, 0.21, 0, 0)
                expect(a).to_equal(0)
                expect(at).to_equal(0)

        def test_imitating_lift_coefficient(self, topic):
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
            a1, at1 = iterate_induction_factors(LSR, section, solidity, a, 0)
            expect(a).to_almost_equal(a1)
            assert at1 > 0
            # XXX could check at better

    class solve_induction_factors_test:
        def test_it_converges_with_reasonable_input_values(self, topic):
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
            a, at = solve_induction_factors(LSR, section, solidity)
            assert a  > 0 and a  < 0.4
            assert at > 0 and at < 0.2

        def test_it_gives_up_with_unreasonable_input_values(self, topic):
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
                a, at = solve_induction_factors(LSR, section, solidity)


@Vows.batch
class BEMModelTests(Vows.Context):

    class AnnulusWidths:
        root_length = 10

        def topic(self):
            class MockBlade:
                radii = array([0, 2, 4, 6])
                chord = array([1, 1, 1, 1])
                twist = array([1, 1, 1, 1])
                thickness = array([1, 1, 1, 1])
            model = BEMModel(MockBlade(), self.root_length,
                             3, Mock(), unsteady=True)
            return model

        def annuli_edges_are_correct(self, model):
            edges = [annulus.edge_radii for annulus in model.annuli]
            expect(edges).to_equal([
                (10, 11),
                (11, 13),
                (13, 15),
                (15, 16),
            ])
