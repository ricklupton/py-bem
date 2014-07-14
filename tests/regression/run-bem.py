#!/usr/bin/env python

from numpy import pi, array, r_
from bem.bem import BEMModel, AerofoilDatabase
import yaml
import sys


class BladeDef:
    def __init__(self, filename):
        with open(filename, 'rt') as f:
            bdata = [array([float(x) for x in row.split()]) for row in f]
        self.radii, self.chord, self.twist, self.thickness = bdata


def build_model(blade_filename, aerofoil_filename):
    # Load blade & aerofoil definitions
    blade = BladeDef(blade_filename)
    db = AerofoilDatabase(aerofoil_filename)
    root_length = 1.25

    # Create BEM model, interpolating to same output radii as Bladed
    return BEMModel(blade, root_length=root_length, num_blades=3,
                    aerofoil_database=db, unsteady=True)


        #     # Same windspeed and rotor speed as the Bladed run
        # windspeed  = 12            # m/s
        # rotorspeed = 22 * (pi/30)  # rad/s


def find_solution(model, windspeed, rotorspeed, pitch, windspeed2=None):
    # Calculate forces and derivatives at different windspeed?
    if windspeed2 is None:
        windspeed2 = windspeed

    # Along blade
    factors = model.solve(windspeed, rotorspeed, pitch)
    forces = model.forces(windspeed2, rotorspeed, pitch, 1.225, factors)
    xdot = model.inflow_derivatives(windspeed2, rotorspeed, pitch,
                                    factors)

    assert len(model.radii) == len(factors) == len(forces) == len(xdot)
    print("\t".join(['r', 'fx', 'fy', 'a', 'at', 'udot', 'utdot']))
    for i in range(len(model.radii)):
        row = r_[model.radii[i], forces[i], factors[i], xdot[i]]
        print("\t".join(["%g" % xx for xx in row]))

    # Overall rotor
    pcoeffs = model.pcoeffs(windspeed2, rotorspeed, pitch)
    print()
    print("Pcoeffs:")
    print()
    print("\t".join(['CT', 'CQ', 'CP']))
    print("\t".join(["%g" % xx for xx in pcoeffs]))


with open(sys.argv[1], 'rt') as f:
    inp = yaml.safe_load(f)

model = build_model(inp['blade'], inp['aerofoil'])
find_solution(model, inp['windspeed'], pi/30 * inp['rotorspeed'],
              pi/180 * inp['pitch'], inp.get('windspeed2', None))
