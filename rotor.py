"""
Rotor model using mbwind
"""

from numpy import pi, dot
from mbwind import rotations, RigidConnection, RigidBody
from mbwind.elements.modal import ModalElementFromFE

class Rotor(object):
    def __init__(self, num_blades, root_length, blade_fe, num_blade_modes=None):
        self.num_blades = num_blades
        self.root_length = root_length
        self.blade_fe = blade_fe

        # Build the elements
        self.roots = []
        self.blades = []
        for ib in range(num_blades):
            R = rotations(('x', ib*2*pi/3), ('y', -pi/2))
            root_offset = dot(R, [root_length, 0, 0])
            root = RigidConnection('root%d' % (ib+1), root_offset, R)
            blade = ModalElementFromFE('blade%d' % (ib+1),
                                       blade_fe, num_blade_modes)
            root.add_leaf(blade)
            self.roots.append(root)
            self.blades.append(blade)

    @property
    def mass(self):
        return self.num_blades * self.blade_modes.mass

    def connect_to(self, parent):
        for root in self.roots:
            parent.add_leaf(root)
