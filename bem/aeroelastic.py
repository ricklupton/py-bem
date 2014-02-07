"""
Put together structural and BEM models
Rick Lupton
22 Jan 2014
"""

import numpy as np
from bem.rotor import Rotor
from mbwind import System, Hinge
from pybladed.data import BladedRun
import mbwind
from mbwind.blade import Blade
from bem.bem import *
from beamfe import BeamFE, interleave


def build_bem_model(blade, root_length):
    # Create BEM model, interpolating to same output radii as Bladed
    db = AerofoilDatabase('aerofoils.npz')
    model = BEMModel(blade, root_length, num_blades=3,
                     aerofoil_database=db, unsteady=True)
    return model


def build_fe_model(blade, root_length, rotor_speed, num_modes, modal_damping):
    # Calculate centrifugal force
    X0 = interleave(root_length + blade.radii, 6)
    N = BeamFE.centrifugal_force_distribution(X0, blade.density)

    # Make FE model. FE twist is in geometric sense, negative to convention.
    fe = BeamFE(blade.radii, blade.density, blade.EA, blade.EI_flap,
                blade.EI_edge, twist=-blade.twist,
                axial_force=rotor_speed**2 * N)
    fe.set_boundary_conditions('C', 'F')

    modal = fe.modal_matrices(4)
    modal.damping[:] = modal_damping
    return modal


def build_structural_system(root_length, modal_fe, rotor_speed):
    rotor = Rotor(3, root_length, modal_fe, pitch=True)
    system = System()
    shaft = Hinge('shaft', [1, 0, 0])
    rotor.connect_to(shaft)
    system.add_leaf(shaft)
    system.setup()
    system.prescribe(shaft, acc=0.0, vel=rotor_speed)
    for b in rotor.pitch_bearings:
        system.prescribe(b, acc=0, vel=0)
    return rotor, system


class AeroelasticModel:
    def __init__(self, blade_definition, root_length, num_modes,
                 rotor_speed, modal_damping):
        self.blade = blade_definition
        self.root_length = root_length
        self.rotor_speed = rotor_speed
        self.air_density = 1.225

        self.modal = build_fe_model(blade_definition, root_length, rotor_speed,
                                    num_modes, modal_damping)
        self.bem = build_bem_model(blade_definition, root_length)
        self.rotor, self.system = build_structural_system(
            root_length, self.modal, rotor_speed)
        self.system.update_kinematics()

    def plot_system(self):
        from mbwind.visual import SystemView
        from matplotlib.pyplot import subplots
        fig, ax = subplots(2, 2, figsize=(8, 8),
                           subplot_kw=dict(aspect=1), sharex=True,
                           sharey=True)
        axes = [ax[0, 1], ax[1, 0], ax[1, 1]]
        views = [SystemView(a, direction, self.system)
                 for a, direction in zip(axes, ['yx', 'xz', 'yz'])]
        ax[1, 1].set_xlabel('Y')
        ax[1, 0].set_ylabel('Z')
        ax[1, 0].set_xlabel('X')
        ax[0, 1].set_ylabel('X')
        for v in views:
            v.update()
            ax[0, 0].set_xlim(-6, 6)
            ax[0, 0].set_ylim(-6, 6)
        return fig

    def _aerostates2factors(self, states, wind_speed):
        ul = states[0::2]
        ut = states[1::2]
        al = ul / wind_speed
        at = ut / self.rotor_speed / self.bem.radii
        return np.c_[al, at]

    def _factors2aerostates(self, factors, wind_speed):
        q_aero = np.zeros(2 * factors.shape[0])
        q_aero[0::2] = factors[:, 0] * wind_speed
        q_aero[1::2] = factors[:, 1] * self.rotor_speed * self.bem.radii
        return q_aero

    def _apply_aero_loading_to_blade(self, blade, wind_speed, pitch, factors):
        # Blade velocities -- ignore axial velocity
        bvel = self.rotor.transform_blade_to_aero(blade.elastic_velocities())
        bvel_factors = bvel[:, 0:2] / wind_speed

        # Distributed forces (transformed into element coordinates)
        aero_forces = np.zeros_like(bvel)
        aero_forces[:, 0:2] = self.bem.forces(wind_speed, self.rotor_speed,
                                              self.air_density, pitch, factors,
                                              bvel_factors)
        beam_forces = self.rotor.transform_aero_to_blade(aero_forces)

        # Apply the forces to the structural system
        blade.loading = beam_forces

    def do_aeroelasticity(self, wind_speed, pitch, q_aero):
        # Calculate aerodynamic state derivatives
        factors = self._aerostates2factors(q_aero, wind_speed)
        qd_aero = self.bem.inflow_derivatives(
            wind_speed, self.rotor_speed, pitch, factors)

        # Apply aerodynamic loading to blades
        for b in self.rotor.blades:
            self._apply_aero_loading_to_blade(b, wind_speed, pitch, factors)

        # Return aerodynamic state derivatives
        return qd_aero.flatten()

    def find_equilibrium(self, wind_speed, pitch):
        # Calculate initial aerodynamic states
        initial_factors = array(self.bem.solve(wind_speed,
                                               self.rotor_speed, pitch))
        q_aero = self._factors2aerostates(initial_factors, wind_speed)

        # Set structural pitch
        for bearing in self.rotor.pitch_bearings:
            bearing.xstrain[:] = pitch

        # Set blade loading with no velocities
        self.system.qd[:] = 0
        self.do_aeroelasticity(wind_speed, q_aero)

        # Find equilibrium
        self.system.find_equilibrium()
        return self.system.q.dofs[:], q_aero

    def integrate(self, t1, dt, wind_speed_func, pitch=0.0,
                  print_progress=True):

        initial_dofs, initial_aero = self.find_equilibrium(wind_speed_func(0),
                                                           pitch)
        self.system.q.dofs[:] = initial_dofs

        def callback(system, t, q_aero):
            windspeed = wind_speed_func(t)
            qd_aero = self.do_aeroelasticity(windspeed, q_aero, pitch)
            return qd_aero

        integrator = mbwind.Integrator(self.system, outputs=('pos', 'vel'))
        integrator.add_output(mbwind.LoadOutput('node-0', local=False))
        integrator.add_output(mbwind.LoadOutput('node-1', local=True))
        integrator.add_output(self.rotor.blades[0].output_deflections())
        integrator.integrate(t1, dt, callback=callback,
                             extra_states=initial_aero,
                             nprint=(10 if print_progress else None))

        if print_progress:
            print("\nResult contains the following outputs:")
            for i, label in enumerate(integrator.labels + ["aero states"]):
                print("[{:2}] {}".format(i, label))

        return integrator  # contains results
