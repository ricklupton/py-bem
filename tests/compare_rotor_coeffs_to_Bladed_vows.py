from pyvows import Vows, expect
import __init__

from numpy import pi, array, r_
from mbwind.blade import Blade
from pybladed.data import BladedRun

from bem import BEMModel, AerofoilDatabase


# De-duplicate repeated stations in Bladed output
def dedup(x):
    return r_[x[::2], x[-1:]]


demo_a_blade = """0	1.14815	3.44444	5.74074	9.18519	16.0741	26.4074	35.5926	38.2333	38.75
2.06667	2.06667	2.75556	3.44444	3.44444	2.75556	1.83704	1.14815	0.688889	0.028704
0	0	9	13	11	7.800002	3.3	0.3	2.75	4
21	21	21	21	21	21	15	13	13	13"""


@Vows.batch
class BEMModel_comparison_against_Bladed(Vows.Context):

    class Testing_aeroinfo:

        def topic(self):

            # Load reference results from Bladed
            br = BladedRun('../demo_a/aeroinfo_no_tip_loss_root21pc')
            self.bladed_r = dedup(br.find('axiala').x())
            self.bladed = lambda chan: dedup(br.find(chan).get())

            # Load blade & aerofoil definitions
            blade = Blade('../demo_a/aeroinfo_no_tip_loss_root21pc.$PJ')
            db = AerofoilDatabase('../aerofoils.npz')
            root_length = 1.25

            # Create BEM model, interpolating to same output radii as Bladed
            model = BEMModel(blade, root_length=root_length,
                             num_blades=3, aerofoil_database=db,
                             bem_radii=self.bladed_r)
            return model

        def test_loading(self, model):
            # Expected values from Bladed
            bdata = [array(map(float, row.split()))
                     for row in demo_a_blade.split('\n')]
            radii, chord, twist, thickness = bdata
            radii += model.root_length

            expect(radii).to_almost_equal(
                [a.radius for a in model.annuli], atol=1e-3)
            expect(chord).to_almost_equal(
                [a.blade_section.chord for a in model.annuli], atol=1e-3)
            expect(twist).to_almost_equal(
                [a.blade_section.twist * 180/pi for a in model.annuli], atol=1e-1)

            def thick_from_name(name):
                if name[3] == '%': return float(name[:3])
                elif name[2] == '%': return float(name[:2])
                else: raise ValueError('no thickness in name')
            expect(thickness).to_almost_equal(
                [thick_from_name(a.blade_section.foil.name)
                 for a in model.annuli])

        def test_solution_against_Bladed(self, model):
            # Same windspeed and rotor speed as the Bladed run
            windspeed  = 12            # m/s
            rotorspeed = 22 * (pi/30)  # rad/s
            factors = model.solve(windspeed, rotorspeed)
            a, at = zip(*factors)

            # Bladed results
            ba  = self.bladed('axiala')
            bat = self.bladed('tanga')

            expect(a).to_almost_equal(ba,   rtol=1e-2, atol=1e-2)
            # first station doesn't work for some reason... XXX
            expect(at[1:]).to_almost_equal(bat[1:], rtol=1e-2, atol=1e-2)

        def test_forces_against_Bladed(self, model):
            # Same windspeed and rotor speed as the Bladed run
            windspeed  = 12            # m/s
            rotorspeed = 22 * (pi/30)  # rad/s
            forces = model.forces(windspeed, rotorspeed, rho=1.225)
            fx, fy = map(array, zip(*forces))

            # Bladed results
            bfx = self.bladed('dfout')
            bfy = self.bladed('dfin')

            expect(fx / abs(fx).max()).to_almost_equal(bfx / abs(fx).max(), rtol=1e-2, atol=1e-3)
            expect(fy / abs(fy).max()).to_almost_equal(bfy / abs(fy).max(), rtol=1e-2, atol=1e-3)

    class BEMModel_Test_pcoeffs:

        def topic(self):
            # Load reference results from Bladed
            br = BladedRun('../demo_a/pcoeffs_no_tip_loss_root21pc')
            self.bladed_TSR = br.find('pocoef').x()
            self.bladed = lambda chan: br.find(chan).get()

            # Load blade & aerofoil definitions
            blade = Blade('../demo_a/pcoeffs_no_tip_loss_root21pc.$PJ')
            db = AerofoilDatabase('../aerofoils.npz')
            root_length = 1.25

            # Create BEM model, interpolating to same output radii as Bladed
            model = BEMModel(blade, root_length=root_length,
                             num_blades=3, aerofoil_database=db)
            return model

        def test_coefficients_against_Bladed(self, model):
            # Same rotor speed and TSRs as the Bladed run
            rotorspeed = 22 * (pi/30)  # rad/s

            # Don't do every TSR -- just a selection
            nth = 4
            TSRs = self.bladed_TSR[::nth]
            R = model.annuli[-1].radius
            windspeeds = (R*rotorspeed/TSR for TSR in TSRs)
            coeffs = [model.pcoeffs(ws, rotorspeed) for ws in windspeeds]
            CT, CQ, CP = zip(*coeffs)

            # Bladed results
            bCT = self.bladed('thcoef')[::nth]
            bCQ = self.bladed('tocoef')[::nth]
            bCP = self.bladed('pocoef')[::nth]

            expect(CT).to_almost_equal(bCT, rtol=1e-2, atol=1e-2)
            expect(CQ).to_almost_equal(bCQ, rtol=1e-2, atol=1e-2)
            expect(CP).to_almost_equal(bCP, rtol=1e-2, atol=1e-1)
