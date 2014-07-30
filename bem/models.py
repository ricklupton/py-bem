import numpy as np


class FrozenWakeAerodynamics:
    """Calculate induced flows once in given initial conditions"""

    def __init__(self, bem_model, initial_wind_speed,
                 initial_rotor_speed, initial_pitch_angle):
        self.bem_model = bem_model
        # Find the frozen wake state
        self.wake_state = bem_model.solve_wake(initial_wind_speed,
                                               initial_rotor_speed,
                                               initial_pitch_angle)

    def forces(self, wind_speed, rotor_speed, pitch_angle, rho):
        shape_test = (np.asarray(wind_speed) *
                      np.asarray(rotor_speed) *
                      np.asarray(pitch_angle))
        if shape_test.ndim == 0:
            # Single value
            factors = self.wake_state / [wind_speed, rotor_speed]
            factors[:, 1] /= self.bem_model.radii
            forces = self.bem_model.forces(wind_speed, rotor_speed,
                                           pitch_angle, rho, factors)
        elif shape_test.ndim == 1:
            # Multiple values
            inputs = np.zeros((len(shape_test), 3))
            inputs[:, 0] = wind_speed
            inputs[:, 1] = rotor_speed
            inputs[:, 2] = pitch_angle
            forces = np.zeros((inputs.shape[0], self.wake_state.shape[0], 2))
            for i in range(forces.shape[0]):
                factors = self.wake_state / inputs[i, :2]
                factors[:, 1] /= self.bem_model.radii
                forces[i] = self.bem_model.forces(*inputs[i], rho=rho,
                                                  factors=factors)
        else:
            raise ValueError("Bad input shapes: {}".format(shape_test.shape))
        return forces


class EquilibriumWakeAerodynamics:
    """Calculate induced flow for each requested set of conditions"""

    def __init__(self, bem_model):
        self.bem_model = bem_model

    def forces(self, wind_speed, rotor_speed, pitch_angle, rho):
        shape_test = (np.asarray(wind_speed) *
                      np.asarray(rotor_speed) *
                      np.asarray(pitch_angle))
        if shape_test.ndim == 0:
            # Single value
            wake_state = self.bem_model.solve_wake(wind_speed,
                                                   rotor_speed,
                                                   pitch_angle)
            factors = wake_state / [wind_speed, rotor_speed]
            factors[:, 1] /= self.bem_model.radii
            forces = self.bem_model.forces(wind_speed, rotor_speed,
                                           pitch_angle, rho, factors)
        elif shape_test.ndim == 1:
            # Multiple values
            inputs = np.zeros((len(shape_test), 3))
            inputs[:, 0] = wind_speed
            inputs[:, 1] = rotor_speed
            inputs[:, 2] = pitch_angle
            forces = np.zeros((inputs.shape[0], len(self.bem_model.radii), 2))
            for i in range(forces.shape[0]):
                wake_state = self.bem_model.solve_wake(*inputs[i])
                factors = wake_state / inputs[i, :2]
                factors[:, 1] /= self.bem_model.radii
                forces[i] = self.bem_model.forces(*inputs[i], rho=rho,
                                                  factors=factors)
        else:
            raise ValueError("Bad input shapes: {}".format(shape_test.shape))
        return forces
