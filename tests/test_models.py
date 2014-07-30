import unittest
import numpy as np
from numpy import pi
from numpy.testing import assert_allclose
from bem.models import FrozenWakeAerodynamics, EquilibriumWakeAerodynamics
from . import utils


class TestFrozenWakeAerodynamics(unittest.TestCase):
    def setUp(self):
        self.bem = utils.get_test_model()

    def test_calculated_forces_for_frozen_wake_state(self):
        # Initial and later conditions
        args0 = (12.2, 12*pi/30, 0.0)
        args1 = (13.1, 10*pi/30, 2.3)

        # Expected result; NB *wake* is frozen, not *factors*
        factors = self.bem.solve(*args0)
        factors[:, 0] *= args0[0] / args1[0]   # correct for changing mean wind
        factors[:, 1] *= args0[1] / args1[1]   # correct for changing rotor spd
        expected_forces = self.bem.forces(*args1, rho=1.225, factors=factors)

        # Test
        model = FrozenWakeAerodynamics(self.bem, *args0)
        forces = model.forces(*args1, rho=1.225)
        assert_allclose(forces, expected_forces)

    def test_broadcasting(self):
        # Initial and later conditions
        args0 = (12.2, 12*pi/30, 0.0)
        scaling = np.arange(0.8, 1.21, 0.5)

        # Vary each parameter
        for j in range(3):
            # Expected result; NB *wake* is frozen, not *factors*
            expected = np.zeros((len(scaling), len(self.bem.radii), 2))
            for i in range(len(scaling)):
                factors = self.bem.solve(*args0)
                args = list(args0)
                args[j] *= scaling[i]

                # correct for changing mean wind and rotor speed
                factors[:, 0] *= args0[0] / args[0]
                factors[:, 1] *= args0[1] / args[1]

                expected[i] = self.bem.forces(*args, rho=1.225,
                                              factors=factors)

            # Test
            model = FrozenWakeAerodynamics(self.bem, *args0)
            args = list(args0)
            args[j] = scaling * args0[j]
            forces = model.forces(*args, rho=1.225)
            assert_allclose(forces, expected, rtol=1e-6)

    def test_bad_shapes(self):
        # Initial and later conditions
        args0 = (12.2, 12*pi/30, 0.0)
        model = FrozenWakeAerodynamics(self.bem, *args0)
        with self.assertRaises(ValueError):
            model.forces(np.random.random((4, 2)),
                         args0[1], args0[2], rho=1.225)


class TestEquilibriumWakeAerodynamics(unittest.TestCase):
    def setUp(self):
        self.bem = utils.get_test_model()

    def test_calculated_forces_for_equilibrium_wake_state(self):
        # Initial conditions not used. Later conditions:
        args1 = (13.1, 10*pi/30, 2.3)

        # Expected result - wake solved for final state
        factors = self.bem.solve(*args1)
        expected_forces = self.bem.forces(*args1, rho=1.225, factors=factors)

        # Test
        model = EquilibriumWakeAerodynamics(self.bem)
        forces = model.forces(*args1, rho=1.225)
        assert_allclose(forces, expected_forces)

    def test_broadcasting(self):
        # Initial and later conditions
        args0 = (12.2, 12*pi/30, 0.0)
        scaling = np.arange(0.8, 1.21, 0.5)

        # Vary each parameter
        for j in range(3):
            # Expected result; NB *wake* is frozen, not *factors*
            expected = np.zeros((len(scaling), len(self.bem.radii), 2))
            for i in range(len(scaling)):
                args = list(args0)
                args[j] *= scaling[i]
                factors = self.bem.solve(*args)
                expected[i] = self.bem.forces(*args, rho=1.225,
                                              factors=factors)

            # Test
            model = EquilibriumWakeAerodynamics(self.bem)
            args = list(args0)
            args[j] = scaling * args0[j]
            forces = model.forces(*args, rho=1.225)
            assert_allclose(forces, expected, rtol=1e-5)

    def test_bad_shapes(self):
        # Initial and later conditions
        args0 = (12.2, 12*pi/30, 0.0)
        model = EquilibriumWakeAerodynamics(self.bem)
        with self.assertRaises(ValueError):
            model.forces(np.random.random((4, 2)),
                         args0[1], args0[2], rho=1.225)


if __name__ == '__main__':
    unittest.main()
