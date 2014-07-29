import unittest
from numpy import pi, array, r_
from numpy.testing import (assert_array_almost_equal,
                           assert_allclose)

from bem import (BEMModel, AerofoilDatabase)
from mbwind.blade import Blade
from pybladed.data import BladedRun


demo_a_blade = """0	1.14815	3.44444	5.74074	9.18519	16.0741	26.4074	35.5926	38.2333	38.75
2.06667	2.06667	2.75556	3.44444	3.44444	2.75556	1.83704	1.14815	0.688889	0.028704
0	0	9	13	11	7.800002	3.3	0.3	2.75	4
21	21	21	21	21	21	15	13	13	13"""


def get_bdata():
    return [array([float(x) for x in row.split()])
            for row in demo_a_blade.split('\n')]


# De-duplicate repeated stations in Bladed output
def dedup(x):
    return r_[x[::2], x[-1:]]


class BEMModel_aeroinfo_base:
    def setUp(self):
        # Load reference results from Bladed
        br = BladedRun(self.BLADED_RUN)
        self.bladed_r = dedup(br.find('axiala').x())
        self.bladed = lambda chan: dedup(br.find(chan).get())

        # Load blade & aerofoil definitions
        blade = Blade('tests/data/Bladed_demo_a_modified/aeroinfo.$PJ')
        db = AerofoilDatabase('tests/data/aerofoils.npz')
        root_length = 1.25

        # Create BEM model, interpolating to same output radii as Bladed
        self.model = BEMModel(blade, root_length=root_length,
                              num_blades=3, aerofoil_database=db,
                              radii=self.bladed_r)

    def test_loading(self):
        # Expected values from Bladed
        radii, chord, twist, thickness = get_bdata()
        radii += self.model.root_length

        assert_array_almost_equal(radii, self.model.radii, decimal=3)
        assert_array_almost_equal(chord, self.model.chord, decimal=4)
        assert_array_almost_equal(twist, self.model.twist * 180/pi,
                                  decimal=2)

        def thick_from_name(name):
            if name[3] == '%':
                return float(name[:3])
            elif name[2] == '%':
                return float(name[:2])
            else:
                raise ValueError('no thickness in name')
        assert_array_almost_equal(thickness, self.model.thick, decimal=3)

    def test_solution_against_Bladed(self):
        factors = self.model.solve(*self.ARGS)
        a, at = zip(*factors)

        # Bladed results
        ba = self.bladed('axiala')
        bat = self.bladed('tanga')

        assert_array_almost_equal(a, ba, decimal=2)
        assert_array_almost_equal(at, bat, decimal=2)

    def test_forces_against_Bladed(self):
        # Same windspeed and rotor speed as the Bladed run
        factors = self.model.solve(*self.ARGS)
        forces = self.model.forces(*self.ARGS, rho=1.225, factors=factors)
        mfx, mfy = map(array, zip(*forces))

        # Bladed results
        bfx = self.bladed('dfout')
        bfy = self.bladed('dfin')

        assert_array_almost_equal(mfx / abs(mfx).max(),
                                  bfx / abs(mfx).max(), decimal=3)
        assert_array_almost_equal(mfy / abs(mfy).max(),
                                  bfy / abs(mfy).max(), decimal=2)

    def test_solve_finds_equilibrium_solution(self):
        factors = self.model.solve(*self.ARGS)
        xdot = self.model.inflow_derivatives(*self.ARGS, factors=factors)
        assert_allclose(xdot, 0, atol=1e-4)

    def test_inflow_derivatives_with_one_annulus(self):
        factors = self.model.solve(*self.ARGS)
        all_derivs = self.model.inflow_derivatives(
            *self.ARGS, factors=factors*1.1)
        one_derivs = self.model.inflow_derivatives(
            *self.ARGS, factors=factors[7:8]*1.1, annuli=slice(7, 8))
        assert_array_almost_equal(one_derivs, all_derivs[7:8, :])


class BEMModel_Test_aeroinfo_12ms(BEMModel_aeroinfo_base, unittest.TestCase):
    BLADED_RUN = 'tests/data/Bladed_demo_a_modified/aeroinfo'
    ARGS = (
        12.0,           # windspeed [m/s]
        22 * (pi/30),   # rotor speed [rad/s]
        0 * (pi/180),  # pitch angle [rad]
    )


class BEMModel_Test_aeroinfo_14ms(BEMModel_aeroinfo_base, unittest.TestCase):
    BLADED_RUN = 'tests/data/Bladed_demo_a_modified/aeroinfo_14ms_2deg'
    ARGS = (
        14.0,           # windspeed [m/s]
        22 * (pi/30),   # rotor speed [rad/s]
        2 * (pi/180),  # pitch angle [rad]
    )


class BEMModel_Test_pcoeffs(unittest.TestCase):
    def setUp(self):
        # Load reference results from Bladed
        from pybladed.data import BladedRun
        br = BladedRun('tests/data/Bladed_demo_a_modified/pcoeffs')
        self.bladed_TSR = br.find('pocoef').x()
        self.bladed = lambda chan: br.find(chan).get()

        # Load blade & aerofoil definitions
        blade = Blade('tests/data/Bladed_demo_a_modified/pcoeffs.$PJ')
        db = AerofoilDatabase('tests/data/aerofoils.npz')
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
        R = self.model.radii[-1]
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
