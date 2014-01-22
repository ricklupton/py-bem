"""
Rotor model using mbwind
"""

from numpy import pi, dot
from mbwind import rotations, RigidConnection, RigidBody, Hinge
from mbwind.elements.modal import ModalElementFromFE

class Rotor(object):
    def __init__(self, num_blades, root_length, blade_fe, pitch=False):
        self.num_blades = num_blades
        self.root_length = root_length
        self.blade_fe = blade_fe

        # Build the elements
        self.roots = []
        self.blades = []
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
