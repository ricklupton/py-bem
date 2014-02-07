"""
Rotor model using mbwind
"""

import numpy as np
from numpy import pi, dot
from mbwind import rotations, RigidConnection, RigidBody, Hinge
from mbwind.elements.modal import ModalElementFromFE


def transform_rows(A, y):
    return np.einsum('ij, aj -> ai', A, y)


class Rotor(object):
    def __init__(self, num_blades, root_length, blade_fe, pitch=False):
        self.num_blades = num_blades
        self.root_length = root_length
        self.blade_fe = blade_fe

        # Build the elements
        self.roots = []
        self.blades = []
        self.pitch_bearings = []
        for ib in range(num_blades):
            R = rotations(('y', -pi/2), ('x', ib*2*pi/3))
            root_offset = dot(R, [root_length, 0, 0])
            root = RigidConnection('root%d' % (ib+1), root_offset, R)
            blade = ModalElementFromFE('blade%d' % (ib+1), blade_fe)
            self.roots.append(root)
            self.blades.append(blade)

            if pitch:
                # Add bearing about blade X axis
                bearing = Hinge('pitch%d' % (ib+1), [1, 0, 0])
                self.pitch_bearings.append(bearing)
                root.add_leaf(bearing)
                bearing.add_leaf(blade)
            else:
                root.add_leaf(blade)

    @property
    def mass(self):
        return self.num_blades * self.blade_fe.fe.mass

    def connect_to(self, parent):
        for root in self.roots:
            parent.add_leaf(root)

    def transform_blade_to_aero(self, y):
        A = rotations(('y', -pi/2))
        return transform_rows(A, y)

    def transform_aero_to_blade(self, y):
        A = rotations(('y', pi/2))
        return transform_rows(A, y)
