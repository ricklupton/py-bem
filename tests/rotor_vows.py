#from nose.tools import *
#from mock import Mock
from pyvows import Vows, expect
import __init__

from numpy import pi, array, zeros, ones_like, r_, arange, sin, cos, diag, eye

import mbwind
from mbwind.modes import ModalRepresentation
from rotor import Rotor


BLADE_LENGTH = 20.0
def _mock_rigid_uniform_modes(density):
    x = arange(0, BLADE_LENGTH + .1)  # 0 to 20 m inclusive
    blade = ModalRepresentation(x, density * ones_like(x))
    return blade


@Vows.batch
class RotorTests(Vows.Context):

    class WithRigidUniformBlades:
        linear_density = 5.0
        root_length = 1.25

        def topic(self):
            blade = _mock_rigid_uniform_modes(self.linear_density)
            rotor = Rotor(3, self.root_length, blade)
            return rotor

        def it_has_three_blades(self, rotor):
            expect(rotor.num_blades).to_equal(3)

        def it_has_the_right_root_length(self, rotor):
            expect(rotor.root_length).to_equal(1.25)

        def it_has_the_right_rotor_mass(self, rotor):
            blade_mass = self.linear_density * BLADE_LENGTH
            expect(rotor.mass).to_equal(3 * blade_mass)

        class AssembledIntoASystem:
            def topic(self, rotor):
                system = mbwind.System()
                joint = mbwind.FreeJoint('joint')
                system.add_leaf(joint)
                rotor.connect_to(joint)
                system.setup()
                system.update_kinematics()
                return system

            def has_2_elements_per_blade_plus_1_joint(self, system):
                expect(system.elements).to_length(1 + (2 * 3))

            def roots_placed_correctly(self, system):
                # Nodes at end of root segments should be spread round circle
                # in the yz plane by 120 deg, first one on the z axis
                unitvec = lambda i: array([0, -sin(2*i*pi/3), cos(2*i*pi/3)])
                rotor = self.parent.topic_value
                r0 = self.parent.root_length
                r1 = self.parent.root_length + BLADE_LENGTH
                for i in range(3):
                    stns = rotor.blades[i].station_positions()
                    expect(stns[ 0]).to_almost_equal(r0 * unitvec(i))
                    expect(stns[-1]).to_almost_equal(r1 * unitvec(i))

            class ItsMassMatrix:
                def topic(self, system):
                    rsys = mbwind.ReducedSystem(system)
                    return rsys.M

                def is_correct(self, M):
                    linear_density = self.parent.parent.linear_density
                    root_length = self.parent.parent.root_length
                    blade_mass = linear_density * BLADE_LENGTH

                    # Expected values: perp inertia using projected lengths
                    Iy = blade_mass * (BLADE_LENGTH**2 / 12 +
                                       (BLADE_LENGTH/2 + root_length)**2)
                    Iperp = Iy + Iy/4 + Iy/4
                    Iaxial = 3 * Iy
                    expected_mass = 3 * blade_mass * eye(3)
                    expected_inertia = diag([Iaxial, Iperp, Iperp])
                    expected_offdiag = zeros((3, 3))

                    expect(M.shape).to_equal((6, 6))
                    expect(M[:3, :3]).to_almost_equal(expected_mass)
                    expect(M[3:, 3:]).to_almost_equal(expected_inertia)
                    expect(M[3:, :3]).to_almost_equal(expected_offdiag)
                    expect(M[:3, 3:]).to_almost_equal(expected_offdiag.T)

