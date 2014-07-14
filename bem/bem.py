import numpy as np
from numpy import pi, sin, cos, arctan2, trapz, array, dot
from scipy.interpolate import interp1d


def strip_boundaries(radii):
    # Find two ends of strip -- halfway between this point and
    # neighbours, apart from at ends when it's half as wide.
    radii = 1.0 * np.asarray(radii)
    midpoints = (radii[1:] + radii[:-1]) / 2
    return np.r_[radii[0], midpoints, radii[-1]]


def wrap_angle(theta):
    """Wraps the angle to [-pi, pi]"""
    return (theta + pi) % (2 * pi) - pi


class AerofoilDatabase(object):
    def __init__(self, filename):
        self.filename = filename
        self.aerofoils = np.load(filename)

        # Reinterpolate data for all aerofoils to consistent values of alpha
        datasets = self.aerofoils['datasets']
        alpha = []
        for a in sorted(a for data in datasets for a in data['alpha']):
            if alpha and abs(a - alpha[-1]) < 1e-5:
                continue
            alpha.append(a)
        self.alpha = np.array(alpha)
        lift_drag = np.dstack([
            [np.interp(alpha, data['alpha'], data['CL']) for data in datasets],
            [np.interp(alpha, data['alpha'], data['CD']) for data in datasets]
        ])
        self.lift_drag_by_thickness = interp1d(
            self.aerofoils['thicknesses'], lift_drag, axis=0, copy=False)

    # def for_thickness(self, thickness):
    #     lift_drag = self.lift_drag_by_thickness(thickness)
    #     interpolator = interp1d(self.alpha, lift_drag, axis=0)
    #     lift_drag_by_alpha = lambda alpha: interpolator(wrap_angle(alpha))
    #     return Aerofoil('%02d%% thickness' % (100 * thickness),
    #                     lift_drag_by_alpha)

    def for_thickness(self, thickness):
        lift_drag = self.lift_drag_by_thickness(thickness)
        return lift_drag


class Aerofoil(object):
    def __init__(self, name, lift_drag_func):
        self.name = name
        self.lift_drag = lift_drag_func


class BladeSection(object):
    def __init__(self, chord, twist, foil):
        self.chord = chord
        self.twist = twist
        self.foil = foil

    def force_coefficients(self, inflow_angle, pitch):
        # lift & drag coefficients
        alpha = inflow_angle - self.twist - pitch
        cl_cd = self.foil.lift_drag(alpha)

        # resolve in- and out-of-plane
        cphi, sphi = cos(inflow_angle), sin(inflow_angle)
        A = array([[cphi, sphi], [-sphi, cphi]])
        cx_cy = dot(A, cl_cd)
        return cx_cy


def thrust_correction_factor(a):
    """Correction to the thrust for high induction factors"""
    a = np.atleast_1d(a)
    H = np.ones_like(a)
    i = (a > 0.3539)
    H[i] = (4*a*(1-a) / (0.60 + 0.61*a + 0.79*a**2))[i]
    return H


def LSR(windspeed, rotorspeed, radius):
    return radius * rotorspeed / windspeed


def inflow(LSR, factors, extra_velocity_factors=None):
    """Calculate inflow angle from LSR, induction factors and normalised
    extra blade velocities"""
    if extra_velocity_factors is None:
        extra_velocity_factors = np.zeros_like(factors)
    factors = np.atleast_2d(factors)
    extra_velocity_factors = np.atleast_2d(extra_velocity_factors)
    a, at = factors[:, 0], factors[:, 1]
    xdot, ydot = extra_velocity_factors[:, 0], extra_velocity_factors[:, 1]
    Ux = (1.0 - a) - xdot
    Uy = LSR * (1.0 + at) - ydot
    phi = np.arctan2(Ux, Uy)
    inplane = (abs(phi) < 1e-2)
    W = np.zeros_like(phi)
    W[inplane] = Uy[inplane] / cos(phi[inplane])
    W[~inplane] = Ux[~inplane] / sin(phi[~inplane])
    return W, phi


def iterate_induction_factors(LSR, force_coeffs, solidity, pitch,
                              factors, extra_velocity_factors=None):
    a, at = factors[:, 0], factors[:, 1]
    W, phi = inflow(LSR, factors, extra_velocity_factors)
    cx, cy = force_coeffs[:, 0], force_coeffs[:, 1]

    # calculate new induction factors
    # if solidity * cx == 0:
    #     new_a = 0
    # else:
    #     Kx = 4*sin(phi)**2 / (solidity*cx)
    #     H = thrust_correction_factor(a)
    #     new_a = 1. / (Kx/H + 1)

    # if solidity * cy == 0:
    #     new_at = 0
    # else:
    #     Ky = 4*sin(phi)*cos(phi) / (solidity*cy)
    #     new_at = 1. / (-Ky - 1)
    Kx = np.inf * np.ones_like(a)
    Ky = np.inf * np.ones_like(a)
    ix = (solidity * cx != 0)
    iy = (solidity * cy != 0)

    Kx[ix] = 4*sin(phi[ix])**2 / (solidity*cx)[ix]
    H = thrust_correction_factor(a)
    new_a = 1. / (Kx/H + 1)

    Ky[iy] = 4*sin(phi[iy])*cos(phi[iy]) / (solidity*cy)[iy]
    new_at = 1. / (-Ky - 1)

    # Slow down iteration a bit to improve convergence.
    # XXX is there a justification for this?
    new_a = (a + 3*new_a) / 4
    new_at = (at + 3*new_at) / 4

    return np.c_[new_a, new_at]


# def solve_induction_factors(LSR, force_coeffs_func, solidity, pitch,
#                             extra_velocity_factors=None,
#                             tol=None, max_iterations=500, initial=None):
#     """
#     Parameters:
#      - LSR:      local speed ratio = omega r / U
#      - twist:    twist angle of blade
#      - solidity: chord solidity = (B c / 2 pi r)
#      - extra_velocity: blade structural velocities normalised by U
#     """
#     if tol is None:
#         tol = 1e-6
#     if initial is None:
#         a = at = 0
#     else:
#         a, at = initial
#     for i in range(max_iterations):
#         a1, at1 = iterate_induction_factors(LSR, force_coeffs_func,
#                                             solidity, pitch, (a, at),
#                                             extra_velocity_factors)
#         if abs(a1 - a) < tol and abs(at1 - at) < tol:
#             return a1, at1
#         a, at = a1, at1
#     raise RuntimeError("maximum iterations reached")


class BEMAnnulus(object):
    def __init__(self, radius, chord, twist, foil, num_blades):
        self.radius = radius
        self.blade_section = BladeSection(chord, twist, foil)
        self.num_blades = num_blades
        self._last_factors = None

    @property
    def solidity(self):
        return (self.num_blades * self.blade_section.chord /
                (2 * pi * self.radius))

    def solve(self, windspeed, rotorspeed, pitch, extra_velocity_factors=None,
              tol=None):
        a, at = solve_induction_factors(
            LSR(windspeed, rotorspeed, self.radius),
            self.blade_section,
            self.solidity,
            pitch,
            extra_velocity_factors,
            tol,
            initial=self._last_factors)
        self._last_factors = a, at
        return a, at

    def solve_wake(self, windspeed, rotorspeed, pitch, extra_velocities=None,
                   tol=None):
        if extra_velocities is not None:
            extra_velocity_factors = extra_velocities / windspeed
        else:
            extra_velocity_factors = None
        a, at = self.solve(windspeed, rotorspeed, pitch, extra_velocity_factors)
        return (a * windspeed), (at * self.radius * rotorspeed)

    def forces(self, windspeed, rotorspeed, pitch, rho,
               factors, extra_velocity_factors=None):
        """Calculate in- and out-of-plane forces per unit length"""

        # Calculate force coefficients
        Wnorm, phi = inflow(LSR(windspeed, rotorspeed, self.radius),
                            factors, extra_velocity_factors)
        W = windspeed * Wnorm
        cx, cy = self.blade_section.force_coefficients(phi, pitch)

        # De-normalise to actual forces
        fx = 0.5 * rho * W**2 * self.blade_section.chord * cx
        fy = 0.5 * rho * W**2 * self.blade_section.chord * cy

        return fx, fy


class UnsteadyBEMAnnulus(BEMAnnulus):
    def __init__(self, radius, chord, twist, foil, num_blades, edge_radii):
        super(UnsteadyBEMAnnulus, self).__init__(radius, chord, twist, foil, num_blades)
        self.edge_radii = edge_radii

    def inflow_derivatives(self, windspeed, rotorspeed, pitch,
                           factors, extra_velocity_factors=None):
        """Calculate the derivatives of the aerodynamic induced velocities for
        an annulus $$ C_T = 4 a (1-a) + \frac{16}{3 \pi U_0}
        \frac{R_2^3 - R_1^3}{R_2^2 - R_1^2} \dot{a} $$

        """
        u, ut = factors[0] * windspeed, factors[1] * rotorspeed * self.radius
        Wnorm, phi = inflow(LSR(windspeed, rotorspeed, self.radius),
                            factors, extra_velocity_factors)
        cx, cy = self.blade_section.force_coefficients(phi, pitch)
        Kx = 4 * sin(phi) ** 2 / (cx * self.solidity)
        Ky = 4 * sin(phi) * cos(phi) / (self.solidity * cy)

        R1, R2 = self.edge_radii
        mu = (16.0 / (3*pi)) * (R2**3 - R1**3) / (R2**2 - R1**2)

        H = thrust_correction_factor(factors[0])
        if abs(factors[0] - 1) < 1e-3:
            udot = -factors[0] * (0.60*windspeed**2
                                  + 0.61*u*windspeed + 0.79*u**2) / mu
            utdot = 0
        else:
            udot = 4 * (windspeed - u) * ((windspeed - u) / Kx - (u / H)) / mu
            utdot = 4 * (windspeed - u) * (
                -(rotorspeed * self.radius + ut) / Ky - ut) / mu
        return udot, utdot


class BEMModel(object):
    def __init__(self, blade, root_length, num_blades, aerofoil_database,
                 radii=None, unsteady=False):

        if radii is None:
            radii = root_length + blade.radii

        self.blade = blade
        self.root_length = root_length
        self.num_blades = num_blades
        self.unsteady = unsteady

        self.radii = radii
        self.boundaries = strip_boundaries(radii)
        self.chord = np.interp(radii, root_length + blade.radii, blade.chord)
        self.twist = np.interp(radii, root_length + blade.radii, blade.twist)
        self.thick = np.interp(radii, root_length + blade.radii,
                               blade.thickness)
        self.solidity = self.num_blades * self.chord / (2 * pi * radii)

        # Aerofoil data
        self.alpha = aerofoil_database.alpha
        self.lift_drag_data = np.array([
            aerofoil_database.for_thickness(th / 100)
            for th in self.thick])
        self._lift_drag_interp = [
            interp1d(aerofoil_database.alpha,
                     aerofoil_database.for_thickness(th / 100), axis=0)
            for th in self.thick]
        # self.lift_drag = interp1d(aerofoil_database.alpha,
        #                           lift_drag_data, axis=1)
        self._last_factors = np.zeros((len(radii), 2))

    def lift_drag(self, alpha):
        return np.array([f(a) for a, f in zip(alpha, self._lift_drag_interp)])

    def solve(self, windspeed, rotorspeed, pitch,
              extra_velocity_factors=None, tol=None,
              max_iterations=500):

        if tol is None:
            tol = 1e-6

        factors = self._last_factors
        for i in range(max_iterations):
            lsr = LSR(windspeed, rotorspeed, self.radii)
            W, phi = inflow(lsr, factors, extra_velocity_factors)
            force_coeffs = self.force_coefficients(phi, pitch)
            new_factors = iterate_induction_factors(lsr, force_coeffs,
                                                    self.solidity,
                                                    pitch, factors,
                                                    extra_velocity_factors)
            if np.max(abs(new_factors - factors)) < tol:
                self._last_factors = new_factors
                return new_factors
            factors = new_factors
        raise RuntimeError("maximum iterations reached")

    def force_coefficients(self, inflow_angle, pitch):
        # lift & drag coefficients
        alpha = wrap_angle(inflow_angle - self.twist - pitch)
        cl_cd = self.lift_drag(alpha)

        # resolve in- and out-of-plane
        cphi, sphi = np.cos(inflow_angle), np.sin(inflow_angle)
        # A = array([[cphi, sphi], [-sphi, cphi]])
        # cx_cy = dot(A, cl_cd)
        cx_cy = np.c_[
            cl_cd[:, 0] * cphi + cl_cd[:, 1] * sphi,
            cl_cd[:, 0] * -sphi + cl_cd[:, 1] * cphi,
        ]
        return cx_cy

    def solve_wake(self, windspeed, rotorspeed, pitch, extra_velocities=None,
                   tol=None):
        if extra_velocities is not None:
            extra_velocity_factors = extra_velocities / windspeed
        else:
            extra_velocity_factors = None
        factors = self.solve(windspeed, rotorspeed, pitch,
                             extra_velocity_factors)
        factors[:, 0] *= windspeed
        factors[:, 1] *= self.radii * rotorspeed
        return factors

    def inflow_derivatives(self, windspeed, rotorspeed, pitch,
                           factors, extra_velocity_factors=None):
        """Calculate the derivatives of the aerodynamic induced velocities for
        an annuli $$ C_T = 4 a (1-a) + \frac{16}{3 \pi U_0}
        \frac{R_2^3 - R_1^3}{R_2^2 - R_1^2} \dot{a} $$

        """
        u = factors[:, 0] * windspeed
        ut = factors[:, 1] * rotorspeed * self.radii
        Wnorm, phi = inflow(LSR(windspeed, rotorspeed, self.radii),
                            factors, extra_velocity_factors)
        force_coeffs = self.force_coefficients(phi, pitch)
        cx, cy = force_coeffs[:, 0], force_coeffs[:, 1]
        Kx = 4 * sin(phi) ** 2 / (cx * self.solidity)
        Ky = 4 * sin(phi) * cos(phi) / (self.solidity * cy)

        R1, R2 = self.boundaries[:-1], self.boundaries[1:]
        mu = (16.0 / (3*pi)) * (R2**3 - R1**3) / (R2**2 - R1**2)

        H = thrust_correction_factor(factors[:, 0])
        ii = abs(factors[:, 0] - 1) < 1e-3
        udot, utdot = np.zeros_like(u), np.zeros_like(ut)

        # Special case
        udot[ii] = -factors[ii, 0] * (0.60*windspeed**2 +
                                      0.61*u[ii]*windspeed +
                                      0.79*u[ii]**2) / mu[ii]

        # Normal case
        udot[~ii] = (4 * (windspeed - u[~ii]) *
                     ((windspeed - u[~ii]) / Kx[~ii] - (u / H)[~ii]) / mu[~ii])
        utdot[~ii] = (4 * (windspeed - u[~ii]) * (
            -(rotorspeed * self.radii[~ii] + ut[~ii]) / Ky[~ii]
            - ut[~ii]) / mu[~ii])
        return np.c_[udot, utdot]

    def forces(self, windspeed, rotorspeed, pitch, rho,
               factors, extra_velocity_factors=None):
        """Calculate in- and out-of-plane forces per unit length"""

        if extra_velocity_factors is None:
            extra_velocity_factors = np.zeros_like(factors)

        # Calculate force coefficients
        Wnorm, phi = inflow(LSR(windspeed, rotorspeed, self.radii),
                            factors, extra_velocity_factors)
        W = windspeed * Wnorm
        force_coeffs = self.force_coefficients(phi, pitch)
        cx, cy = force_coeffs[:, 0], force_coeffs[:, 1]

        # De-normalise to actual forces
        fx = 0.5 * rho * W**2 * self.chord * cx
        fy = 0.5 * rho * W**2 * self.chord * cy
        forces = np.c_[fx, fy]

        # Force last station to have zero force for compatibility with Bladed
        # XXX this wouldn't work if the last station isn't guaranteed
        #     to be at the tip
        forces[-1] = (0, 0)

        return forces

    def pcoeffs(self, windspeed, rotorspeed, pitch=0.0):
        # We'll nondimensionalise again later so value of rho doesn't matter
        factors = self.solve(windspeed, rotorspeed, pitch)
        forces = self.forces(windspeed, rotorspeed, pitch,
                             rho=1, factors=factors)
        fx, fy = zip(*forces)

        # Integrate forces and moments about shaft
        r = self.radii
        thrust = self.num_blades * trapz(fx, x=r)
        torque = self.num_blades * trapz(-array(fy) * r, x=r)
        power = torque * rotorspeed

        # Nondimensionalise
        A = pi * r[-1]**2
        CT = thrust / (0.5 * 1 * windspeed**2 * A)
        CQ = torque / (0.5 * 1 * windspeed**2 * A * r[-1])
        CP = power  / (0.5 * 1 * windspeed**3 * A)

        return CT, CQ, CP
