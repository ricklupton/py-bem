import numpy as np
from numpy import pi, sin, cos, arctan2, trapz, array, newaxis, nan
from scipy.interpolate import interp1d
from .fast_interpolation import fast_interpolation
import yaml


class Blade:
    """Holds a blade definition.

    Attributes
    ----------
    x : ndarray
        Position of blade stations (measured from blade root)
    chord : ndarray
        Chord length [m]
    twist : ndarray
        Twist, positive points leading edge upwind [rad]
    thickness : ndarray
        Percentage thickness of aerofoil [%]
    density : ndarray, optional
        Mass per unit length of blade [kg/m]
    EA : ndarray, optional
        Axial stiffness
    EI_flap, EI_edge : ndarray, optional
        Bending stiffness in flapwise and edgewise directions

    """
    def __init__(self, x, chord, twist, thickness,
                 density=None, EA=None, EI_flap=None, EI_edge=None):

        self.x = array(x)

        # Optional mass properties
        if density is None:
            density = nan * self.x
        if EA is None:
            EA = nan * self.x
        if EI_flap is None:
            EI_flap = nan * self.x
        if EI_edge is None:
            EI_edge = nan * self.x

        self.chord = array(chord)
        self.twist = array(twist)
        self.thickness = array(thickness)
        self.density = array(density)
        self.EA = array(EA)
        self.EI_flap = array(EI_flap)
        self.EI_edge = array(EI_edge)

        if not (len(x) == len(chord) == len(twist) == len(thickness) ==
                len(density) == len(EA) == len(EI_flap) == len(EI_edge)):
            raise ValueError("Shape mismatch")

    @classmethod
    def from_yaml(cls, filename_or_file):
        """Load blade definition from YAML file.

        The file should have `x`, `chord`, `twist` and `thickness` keys.

        Note
        ----
        NB: In the definition file, the twist is measured in degrees!

        """
        if isinstance(filename_or_file, str):
            with open(filename_or_file) as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(filename_or_file)
        return Blade(data['x'],
                     data['chord'],
                     array(data['twist']) * pi / 180,
                     data['thickness'],
                     data.get('density'),
                     data.get('EA'),
                     data.get('EI_flap'),
                     data.get('EI_edge'))

    def resample(self, new_x):
        return Blade(new_x,
                     np.interp(new_x, self.x, self.chord),
                     np.interp(new_x, self.x, self.twist),
                     np.interp(new_x, self.x, self.thickness),
                     np.interp(new_x, self.x, self.density),
                     np.interp(new_x, self.x, self.EA),
                     np.interp(new_x, self.x, self.EI_flap),
                     np.interp(new_x, self.x, self.EI_edge))


class AerofoilDatabase(object):
    """Store aerofoil list and drag data.

    Loads data in `.npz` format. The data file should have two variables:

    datasets : list of aerofoils
    thicknesses : fractional thicknesses of the aerofoils in `datasets`

    Each aerofoil is an array with `alpha`, `CL`, `CD` and `CM`
    columns, where the angles are in radians.

    """
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

    def for_thickness(self, thickness):
        """Return interpolated lift & drag data for the given thickness.

        Parameters
        ----------
        thickness : float
            Fractional thickness

        """
        lift_drag = self.lift_drag_by_thickness(thickness)
        return lift_drag


def _strip_boundaries(radii):
    # Find two ends of strip -- halfway between this point and
    # neighbours, apart from at ends when it's half as wide.
    radii = 1.0 * np.asarray(radii)
    midpoints = (radii[1:] + radii[:-1]) / 2
    return np.r_[radii[0], midpoints, radii[-1]]


def _wrap_angle(theta):
    """Wraps the angle to [-pi, pi]"""
    return (theta + pi) % (2 * pi) - pi


def _thrust_correction_factor(a):
    """Correction to the thrust for high induction factors"""
    a = np.atleast_1d(a)
    H = np.ones_like(a)
    i = (a > 0.3539)
    ai = a[i]
    H[i] = (4*ai*(1-ai) / (0.60 + 0.61*ai + 0.79*ai**2))
    return H


def LSR(windspeed, rotorspeed, radius):
    return radius * rotorspeed / windspeed


def inflow(LSR, factors, extra_velocity_factors=None):
    """Calculate inflow angle from LSR, induction factors and normalised
    extra blade velocities"""
    factors = np.asarray(factors)
    Ux = (1.0 - factors[:, 0])
    Uy = LSR * (1.0 + factors[:, 1])
    if extra_velocity_factors is not None:
        Ux -= extra_velocity_factors[:, 0]
        Uy -= extra_velocity_factors[:, 1]
    phi = arctan2(Ux, Uy)
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
    Kx = np.inf * np.ones_like(a)
    Ky = np.inf * np.ones_like(a)
    ix = (solidity * cx != 0)
    iy = (solidity * cy != 0)

    Kx[ix] = 4*sin(phi[ix])**2 / (solidity*cx)[ix]
    Ky[iy] = 4*sin(phi[iy])*cos(phi[iy]) / (solidity*cy)[iy]
    H = _thrust_correction_factor(a)

    new = np.empty_like(factors)
    new[:, 0] = 1. / (Kx/H + 1)
    new[:, 1] = 1. / (-Ky - 1)

    # Slow down iteration a bit to improve convergence.
    # XXX is there a justification for this?
    new[...] = (factors + 3*new) / 4
    return new


class BEMModel(object):
    """A Blade Element - Momentum model.

    Parameters
    ----------
    blade : Blade object
        Blade parameter definition
    root_length : float
        Distance from centre of rotor to start of blade
    num_blades : int
        Number of blades in the rotor
    aerofoil_database : AerofoilDatabase object
        Definitions of aerofoil coefficients

    """
    def __init__(self, blade, root_length, num_blades, aerofoil_database):
        self.blade = blade
        self.root_length = root_length
        self.num_blades = num_blades

        self.radii = root_length + np.asarray(self.blade.x)
        self.boundaries = _strip_boundaries(self.radii)
        self.solidity = (self.num_blades * self.blade.chord /
                         (2 * pi * self.radii))

        # Aerofoil data
        self.alpha = aerofoil_database.alpha
        self.lift_drag_data = np.array([
            aerofoil_database.for_thickness(th / 100)
            for th in self.blade.thickness])
        self._lift_drag_interp = fast_interpolation(
            aerofoil_database.alpha, self.lift_drag_data, axis=1)
        self._last_factors = np.zeros((len(self.radii), 2))

    def lift_drag(self, alpha, annuli=None):
        """Interpolate lift & drag coefficients for given angle of attack.

        Parameters
        ----------
        alpha : array_like
            Angle of attach at each annulus [radians]
        annuli : slice or indices, optional
            Subset of annuli to return data for. If given, `alpha`
            should refer only to the annuli of interest.

        Returns
        -------
        Array of shape (number of annuli, 2) containing CL and CD.

        """
        if annuli is None or annuli == slice(None):
            alpha = np.vstack((alpha, alpha)).T
            return self._lift_drag_interp(alpha)
        else:
            data = self.lift_drag_data[annuli]
            if len(alpha) != data.shape[0]:
                raise ValueError("Shape mismatch %s != %s" %
                                 (len(self.alpha), data.shape))
            return np.array([
                interp1d(self.alpha, self.lift_drag_data[annuli][i],
                         axis=-2, copy=False)(alpha[i])
                for i in range(len(alpha))
            ])

    def force_coefficients(self, inflow_angle, pitch, annuli=None):
        """Calculate force coefficients for given inflow.

        The force coefficients Cx and Cy are the out-of-plane and
        in-plane non-dimensional force per unit length, respectively.

        Parameters
        ----------
        inflow_angle : array_like
            Inflow angle at each annulus [radians]. Zero is in-plane, positive
            is towards upwind.
        annuli : slice or indices, optional
            Subset of annuli to return data for. If given, `alpha`
            should refer only to the annuli of interest.

        Returns
        -------
        Array of shape (number of annuli, 2) containing CL and CD.

        """
        if annuli is None:
            annuli = slice(None)
        twist = self.blade.twist[annuli]
        if len(twist) != len(inflow_angle):
            raise ValueError("Shape mismatch")

        # lift & drag coefficients
        alpha = _wrap_angle(inflow_angle - twist - pitch)
        cl_cd = self.lift_drag(alpha, annuli)

        # resolve in- and out-of-plane
        cphi, sphi = np.cos(inflow_angle), np.sin(inflow_angle)
        A = array([[cphi, sphi], [-sphi, cphi]])
        # cx_cy = dot(A, cl_cd)
        # cx_cy = np.c_[
        #     cl_cd[:, 0] * cphi + cl_cd[:, 1] * sphi,
        #     cl_cd[:, 0] * -sphi + cl_cd[:, 1] * cphi,
        # ]
        cx_cy = np.einsum('ijh, hj -> hi', A, cl_cd)
        return cx_cy

    def solve(self, windspeed, rotorspeed, pitch,
              extra_velocity_factors=None, tol=None,
              max_iterations=500, annuli=None):
        """Calculate the BEM solution for the given conditions.

        Parameters
        ----------
        windspeed : float
            Free-stream wind speed
        rotorspeed : float
            Rotor speed [rad/s]
        pitch : float
            Pitch angle [rad]
        extra_velocity_factors : ndarray, optional
            Blade velocity normalised by windspeed
        tol : float, optional
            Absolute tolerance for solution
        max_iterations : int, optional
            Maximum number of iterations
        annuli : slice or indices, optional
            Subset of annuli to return data for.

        Returns
        -------
        Array of axial and tangential induction factors at each annulus,
        shape (number of annuli, 2).

        Raises
        ------
        RuntimeError if maximum number of iterations reached.

        """
        if tol is None:
            tol = 1e-6
        if annuli is None:
            annuli = slice(None)

        r = self.radii[annuli]
        factors = self._last_factors[annuli]
        for i in range(max_iterations):
            lsr = LSR(windspeed, rotorspeed, r)
            W, phi = inflow(lsr, factors, extra_velocity_factors)
            force_coeffs = self.force_coefficients(phi, pitch, annuli)
            new_factors = iterate_induction_factors(lsr, force_coeffs,
                                                    self.solidity[annuli],
                                                    pitch, factors,
                                                    extra_velocity_factors)
            if np.max(abs(new_factors - factors)) < tol:
                self._last_factors[annuli] = new_factors
                return new_factors
            factors = new_factors
        raise RuntimeError("maximum iterations reached")

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
                           factors, extra_velocity_factors=None, annuli=None):
        """Calculate the derivatives of the aerodynamic induced velocities for
        an annuli $$ C_T = 4 a (1-a) + \frac{16}{3 \pi U_0}
        \frac{R_2^3 - R_1^3}{R_2^2 - R_1^2} \dot{a} $$

        """

        if annuli is None:
            annuli = slice(None)

        r = self.radii[annuli]
        u = factors[:, 0] * windspeed
        ut = factors[:, 1] * rotorspeed * r
        if not (r.shape == u.shape == ut.shape):
            raise ValueError("Shape mismatch")

        Wnorm, phi = inflow(LSR(windspeed, rotorspeed, r),
                            factors, extra_velocity_factors)
        force_coeffs = self.force_coefficients(phi, pitch, annuli)
        cx, cy = force_coeffs[:, 0], force_coeffs[:, 1]
        Kx = 4 * sin(phi) ** 2 / (cx * self.solidity[annuli])
        Ky = 4 * sin(phi) * cos(phi) / (self.solidity[annuli] * cy)

        R1, R2 = self.boundaries[:-1][annuli], self.boundaries[1:][annuli]
        mu = (16.0 / (3*pi)) * (R2**3 - R1**3) / (R2**2 - R1**2)

        H = _thrust_correction_factor(factors[:, 0])
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
            -(rotorspeed * r[~ii] + ut[~ii]) / Ky[~ii]
            - ut[~ii]) / mu[~ii])
        return np.c_[udot, utdot]

    def forces(self, windspeed, rotorspeed, pitch, rho,
               factors, extra_velocity_factors=None, annuli=None):
        """Calculate in- and out-of-plane forces per unit length"""

        if extra_velocity_factors is None:
            extra_velocity_factors = np.zeros_like(factors)
        if annuli is None:
            annuli = slice(None)

        factors = np.asarray(factors)
        r = self.radii[annuli]
        chord = self.blade.chord[annuli]
        if not len(r) == factors.shape[0]:
            raise ValueError("Shape mismatch")

        # Calculate force coefficients
        Wnorm, phi = inflow(LSR(windspeed, rotorspeed, r),
                            factors, extra_velocity_factors)
        W = windspeed * Wnorm
        force_coeffs = self.force_coefficients(phi, pitch, annuli)
        forces = (0.5 * rho * W[:, newaxis]**2 *
                  chord[:, newaxis] * force_coeffs)

        # Force last station to have zero force for compatibility with Bladed
        # XXX this wouldn't work if the last station isn't guaranteed
        #     to be at the tip
        if r[-1] == self.radii[-1]:
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
        CP = power / (0.5 * 1 * windspeed**3 * A)

        return CT, CQ, CP
