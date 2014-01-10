from nose.tools import *
from mock import Mock
from numpy import pi, array, r_
from numpy.testing import (assert_array_almost_equal,
                           assert_array_almost_equal_nulp)
from spec import Spec

from bem import *
from mbwind.blade import Blade

class AerofoilDatabase_(Spec):
    def setup(self):
        self.db = AerofoilDatabase('../../aerofoils.npz')

    def test_cylinder_data_is_read_correctly(self):
        foil = self.db.for_thickness(1.0)
        for alpha in [-0.3, 0, 0.3]:
            assert_array_almost_equal(foil.CL(alpha), 0)
            assert_array_almost_equal(foil.CD(alpha), 1)

    def test_thirteen_percent_foil_is_read_correctly(self):
        foil = self.db.for_thickness(0.13)
        assert_array_almost_equal(foil.CL(0), 0.420)
        assert_array_almost_equal(foil.CD(0), 0.006)
        assert_array_almost_equal(foil.CL(10*pi/180), 1.460)
        assert_array_almost_equal(foil.CD(10*pi/180), 0.016)

    def test_interpolation_of_thickness(self):
        # Interpolate between 13% and 17% data
        foil = self.db.for_thickness(0.15)
        assert_array_almost_equal(foil.CL(10*pi/180), (1.460 + 1.500) / 2)
        assert_array_almost_equal(foil.CD(10*pi/180), (0.016 + 0.014) / 2)

    def test_interpolation_of_alpha(self):
        # Interpolate between 4 and 6 deg alpha
        foil = self.db.for_thickness(0.13)
        assert_array_almost_equal(foil.CL(5*pi/180), (0.890 + 1.100) / 2)
        assert_array_almost_equal(foil.CD(5*pi/180), (0.009 + 0.012) / 2)


class BladeSection_Test:
    def holds_chord_twist_and_foil(self):
        sec = BladeSection(1, 2, 3)
        assert_equal(sec.chord, 1)
        assert_equal(sec.twist, 2)
        assert_equal(sec.foil, 3)


class BEM_functions(Spec):

    class thrust_correction_factor_test:
        def test_no_correction_for_small_a(self):
            assert_equal(thrust_correction_factor(0), 1)

        def test_smooth_transition(self):
            a_transition = 0.3539
            H1 = thrust_correction_factor(a_transition)
            H2 = thrust_correction_factor(a_transition + 1e-4)
            assert (H1 - H2) < 1e-3


    class iterate_induction_factors_test:
        def test_there_is_no_induction_if_no_lift_and_drag(self):
            foil = Aerofoil('test', lambda alpha: 0, lambda alpha: 0)
            section = BladeSection(chord=1, twist=0.43, foil=foil)
            a, at = iterate_induction_factors(1, section, 0.21, 0, 0)
            assert_equal(a,  0)
            assert_equal(at, 0)

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
            a1, at1 = iterate_induction_factors(LSR, section, solidity, a, 0)
            assert_equal(a, a1)
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
            a, at = solve_induction_factors(LSR, section, solidity)
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
                a, at = solve_induction_factors(LSR, section, solidity)


demo_a_blade = """0	1.14815	3.44444	5.74074	9.18519	16.0741	26.4074	35.5926	38.2333	38.75
2.06667	2.06667	2.75556	3.44444	3.44444	2.75556	1.83704	1.14815	0.688889	0.028704
0	0	9	13	11	7.800002	3.3	0.3	2.75	4
21	21	21	21	21	21	15	13	13	13"""


class BEMModel_comparison_against_Bladed(Spec):
    class BEMModel_Test_aeroinfo:
        def setup(self):

            # De-duplicate repeated stations in Bladed output
            def dedup(x):
                return r_[x[::2], x[-1:]]

            # Load reference results from Bladed
            from pybladed.data import BladedRun
            br = BladedRun('../../demo_a/aeroinfo_no_tip_loss_root21pc')
            self.bladed_r  = dedup(br.find('axiala').x())
            self.bladed = lambda chan: dedup(br.find(chan).get())

            # Load blade & aerofoil definitions
            blade = Blade('../../demo_a/aeroinfo_no_tip_loss_root21pc.$PJ')
            db = AerofoilDatabase('../../aerofoils.npz')
            root_length = 1.25

            # Create BEM model, interpolating to same output radii as Bladed
            self.model = BEMModel(blade, root_length=root_length,
                                  num_blades=3, aerofoil_database=db,
                                  bem_radii=self.bladed_r)


        def test_loading(self):
            # Expected values from Bladed
            bdata = [array(map(float, row.split()))
                     for row in demo_a_blade.split('\n')]
            radii, chord, twist, thickness = bdata
            radii += self.model.root_length

            assert_array_almost_equal(radii, [a.radius for a in self.model.annuli],
                                      decimal=3)
            assert_array_almost_equal(
                chord, [a.blade_section.chord for a in self.model.annuli],
                decimal=4
            )
            assert_array_almost_equal(twist, [a.blade_section.twist * 180/pi
                                              for a in self.model.annuli],
                                      decimal=2)

            def thick_from_name(name):
                if name[3] == '%': return float(name[:3])
                elif name[2] == '%': return float(name[:2])
                else: raise ValueError('no thickness in name')
            assert_array_almost_equal(thickness,
                                      [thick_from_name(a.blade_section.foil.name)
                                       for a in self.model.annuli])

        def test_solution_against_Bladed(self):
            # Same windspeed and rotor speed as the Bladed run
            windspeed  = 12            # m/s
            rotorspeed = 22 * (pi/30)  # rad/s
            factors = self.model.solve(windspeed, rotorspeed)
            a, at = zip(*factors)

            # Bladed results
            ba  = self.bladed('axiala')
            bat = self.bladed('tanga')

            assert_array_almost_equal(a, ba, decimal=2)
            assert_array_almost_equal(at, bat, decimal=2)

        def test_forces_against_Bladed(self):
            # Same windspeed and rotor speed as the Bladed run
            windspeed  = 12            # m/s
            rotorspeed = 22 * (pi/30)  # rad/s
            forces = self.model.forces(windspeed, rotorspeed, rho=1.225)
            fx, fy = map(array, zip(*forces))

            # Bladed results
            bfx = self.bladed('dfout')
            bfy = self.bladed('dfin')

            assert_array_almost_equal(fx  / abs(fx).max(),
                                      bfx / abs(fx).max(), decimal=3)
            assert_array_almost_equal(fy  / abs(fy).max(),
                                      bfy / abs(fy).max(), decimal=2)


    class BEMModel_Test_pcoeffs:
        def setup(self):

            # Load reference results from Bladed
            from pybladed.data import BladedRun
            br = BladedRun('../../demo_a/pcoeffs_no_tip_loss_root21pc')
            self.bladed_TSR  = br.find('pocoef').x()
            self.bladed = lambda chan: br.find(chan).get()

            # Load blade & aerofoil definitions
            blade = Blade('../../demo_a/pcoeffs_no_tip_loss_root21pc.$PJ')
            db = AerofoilDatabase('../../aerofoils.npz')
            root_length = 1.25

            # Create BEM model, interpolating to same output radii as Bladed
            self.model = BEMModel(blade, root_length=root_length,
                                  num_blades=3, aerofoil_database=db)

        def test_coefficients_against_Bladed(self):
            # Same rotor speed and TSRs as the Bladed run
            rotorspeed = 22 * (pi/30)  # rad/s

            # Don't do every TSR -- just a selection
            nth = 4
            TSRs = self.bladed_TSR[::nth]
            R = self.model.annuli[-1].radius
            windspeeds = (R*rotorspeed/TSR for TSR in TSRs)
            coeffs = [self.model.pcoeffs(ws, rotorspeed) for ws in windspeeds]
            CT, CQ, CP = zip(*coeffs)

            # Bladed results
            bCT = self.bladed('thcoef')[::nth]
            bCQ = self.bladed('tocoef')[::nth]
            bCP = self.bladed('pocoef')[::nth]

            assert_array_almost_equal(CT, bCT, decimal=2)
            assert_array_almost_equal(CQ, bCQ, decimal=2)
            assert_array_almost_equal(CP, bCP, decimal=2)
