import unittest
from numpy import pi, array, r_
from numpy.testing import (assert_array_almost_equal,
                           assert_allclose)

from pybladed.data import BladedRun
from . import utils


# De-duplicate repeated stations in Bladed output
def dedup(x):
    return r_[x[::2], x[-1:]]


class BEMModel_aeroinfo_base:
    def setUp(self):
        # Load reference results from Bladed
        br = BladedRun(self.BLADED_RUN)
        self.bladed_r = dedup(br.find('axiala').x())
        self.bladed = lambda chan: dedup(br.find(chan).get())

        # Create BEM model, interpolating to same output radii as Bladed
        self.model = utils.get_test_model(self.bladed_r)

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

        # Create BEM model, interpolating to same output radii as Bladed
        self.model = utils.get_test_model()

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
