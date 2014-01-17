from nose.tools import *
from numpy import pi, array, zeros, ones_like, r_, arange, sin, cos, diag, eye
from numpy.testing import (assert_array_almost_equal,
                           assert_array_almost_equal_nulp)

import mbwind
from beamfe import BeamFE
from bem.rotor import *

assert_aae = assert_array_almost_equal

def _mock_rigid_uniform_modes(density):
    x = arange(0, 20.1)  # 0 to 20 m inclusive
    fe = BeamFE(x, density, 0, 0, 0)
    return fe

class Rotor_Tests:

    def test_it_can_be_created(self):
        blade = _mock_rigid_uniform_modes(0)
        rotor = Rotor(3, 1.25, blade.modal_matrices(0))
        assert_equal(rotor.num_blades,  3)
        assert_equal(rotor.root_length, 1.25)

    def test_it_has_the_right_rotor_mass(self):
        blade = _mock_rigid_uniform_modes(5)
        rotor = Rotor(3, 1.25, blade.modal_matrices(0))
        assert_equal(rotor.mass,  3 * 5 * 20)

    def test_the_rotor_system_can_be_setup(self):
        linear_density = 5
        root_length = 1.25

        blade = _mock_rigid_uniform_modes(linear_density)
        rotor = Rotor(3, root_length, blade.modal_matrices(0))
        system = mbwind.System()
        rotor.connect_to(system)
        system.setup()
        system.update_kinematics()

        # Should have 2 elements per blade
        assert_equal(len(system.elements), 2 * 3)

        # Nodes at end of root segments should be spread round circle
        # in the yz plane by 120 deg, first one on the z axis
        unitvec = lambda i: array([0, -sin(2*i*pi/3), cos(2*i*pi/3)])
        blade_length = 20
        for i in range(3):
            stns = rotor.blades[i].station_positions()
            assert_aae(stns[0], root_length * unitvec(i))
            assert_aae(stns[-1], (root_length + blade_length) * unitvec(i))


    def test_rotor_system_has_correct_reduced_mass_matrix(self):
        linear_density = 5.0
        root_length = 1.25

        blade = _mock_rigid_uniform_modes(linear_density)
        rotor = Rotor(3, root_length, blade.modal_matrices(0))

        system = mbwind.System()
        joint = mbwind.FreeJoint('joint')
        system.add_leaf(joint)
        rotor.connect_to(joint)
        system.setup()
        system.update_kinematics()
        rsys = mbwind.ReducedSystem(system)

        blade_length = 20.0
        blade_mass = linear_density * blade_length

        # Expected values: perp inertia using projected lengths of beams
        Iy = blade_mass * (blade_length**2 / 12 +
                           (blade_length/2 + root_length)**2)
        Iperp = Iy + Iy/4 + Iy/4
        Iaxial = 3 * Iy
        expected_mass = 3 * blade_mass * eye(3)
        expected_inertia = diag([Iaxial, Iperp, Iperp])
        expected_offdiag = zeros((3, 3))

        assert_equal(rsys.M.shape, (6, 6))
        assert_aae(rsys.M[:3, :3], expected_mass)
        assert_aae(rsys.M[3:, 3:], expected_inertia)
        assert_aae(rsys.M[3:, :3], expected_offdiag)
        assert_aae(rsys.M[:3, 3:], expected_offdiag.T)

