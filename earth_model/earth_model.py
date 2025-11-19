#!/usr/bin/env python
# coding=utf8
"""
Support for PREM-like 1D Earth models

"""

import numpy as np

from .peice_poly import PeicewisePolynomial as PP
from .const import R_EARTH, G


class OneDModel:
    def __init__(
        self,
        breakpoints,
        density_params,
        vp_params,
        vs_params,
        q_mu_params,
        q_kappa_params,
        r_earth=R_EARTH,
    ):
        self.r_earth = r_earth

        self.density_poly = PP(density_params, breakpoints)
        self.vp_poly = PP(vp_params, breakpoints)
        self.vs_poly = PP(vs_params, breakpoints)
        self.qk_poly = PP(q_kappa_params, breakpoints)
        self.qm_poly = PP(q_mu_params, breakpoints)

        self.mass_poly = _setup_mass_poly(self.density_poly, breakpoints)
        self.moi_poly = _setup_moi_poly(self.density_poly, breakpoints)
        self.gravity_poly = _setup_gravity_poly(self.density_poly, breakpoints)
        self.pressure_poly = _setup_pressure_poly(self.gravity_poly, self.density_poly)

    def density(self, r, break_down=False):
        """
        Evaluate density in kg/m**3 at radii r (in km)
        """
        return self.density_poly(r, break_down=break_down)

    def vs(self, r, t=1, break_down=False) -> float | np.ndarray:
        """
        Evaluate s-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        return calculate_vs(
            self.vs_poly, r, t=t, qm_poly=self.qm_poly, break_down=break_down
        )

    def vp(self, r, t=1, break_down=False):
        """
        Evaluate p-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        return calculate_vp(
            self.vp_poly,
            r,
            t=t,
            qk_poly=self.qk_poly,
            qm_poly=self.qm_poly,
            vs_poly=self.vs_poly,
            break_down=break_down,
        )

    def qkappa(self, r, break_down=False):
        qk = self.qk_poly(r, break_down=break_down)
        return qk

    def qshear(self, r, break_down=False):
        qm = self.qm_poly(r, break_down=break_down)
        return qm

    def bulk_modulus(self, r):
        """
        Evaluate bulk modulus (in GPa) at radius r (in km)
        """
        return calculate_bulk_modulus(self.vp_poly, self.vs_poly, self.density_poly, r)

    def shear_modulus(self, r):
        """
        Evaluate shear modulus (in GPa) at radius r (in km)
        """
        return calculate_shear_modulus(self.vs_poly, self.density_poly, r)

    def mass(self, r, r_inner=0.0):
        """
        Evaluate mass inside radius r (in km)
        """
        return calculate_mass(self.mass_poly, r, r_inner=r_inner)

    def moment_of_inertia(self, r, r_inner=0.0):
        """
        Evaluate moment of inertia inside radius r (in km)

        Return a tuple of moment of inertia (in kg m^2) and
        the moment of inertia factor (I/MR**2, dimensionless)
        which is 0.4 for a uniform density body, and decreases
        as the core becomes more dense than the crust/mantle.

        """
        return calculate_moi(self.moi_poly, self.mass_poly, r, r_inner=r_inner)

    def gravity(self, r):
        return calculate_gravity(self.gravity_poly, r)

    def grav_potential(self, r):
        """
        Evaluate the gravitational potential at radius r in J/kg
        """
        return calculate_grav_potential(self.mass_poly, r)

    def pressure(self, r):
        """
        Evaluate pressure (in GPa) at radius r (in km)
        """
        return calculate_pressure(self.pressure_poly, r)


def tabulate_model(
    model: OneDModel, min_step: float, outwards: bool = True
) -> np.recarray:
    """
    Return a record array representing the model handling discontiuities

    This method creates a numpy record array with the model evaulated
    at all depths with a minimum spacing of min_step km. All breakpoints
    are also included in the output. If the densioty is discontinuoius,
    the depth is represented twice, first with the value above the
    discontiuity, then with the value below it. This representation can
    be used to construct travel time curves (for examople).

    The record array contains fields:

        depth (in km)
        radius (in km)
        density (in kg/m^3)
        qkappa (dimensionless quality factor)
        qshear (dimensionless quality factor)

    If outwards=True, element 0 is at the centre of the planet and element -1
    is at the surface. If outwards=False, element 0 is at the surface and
    element -1 is at the centre of the planet.
    """
    # Keep the data as we get it
    radii = np.array([])
    depths = np.array([])
    densities = np.array([])
    vps = np.array([])
    vss = np.array([])
    qks = np.array([])
    qms = np.array([])

    nbps = len(model.density_poly.breakpoints) - 1
    for i in range(nbps):
        j = i if outwards else nbps - i
        k = j + 1 if outwards else j - 1
        rs = np.arange(
            model.density_poly.breakpoints[j],
            model.density_poly.breakpoints[k],
            min_step if outwards else -min_step,
        )
        ds = model.r_earth - rs
        dens = model.density(rs, break_down=not outwards)
        vp = model.vp(rs, break_down=not outwards)
        vs = model.vs(rs, break_down=not outwards)
        qk = model.qkappa(rs, break_down=not outwards)
        qm = model.qshear(rs, break_down=not outwards)
        radii = np.append(radii, rs)
        depths = np.append(depths, ds)
        densities = np.append(densities, dens)
        vps = np.append(vps, vp)
        vss = np.append(vss, vs)
        qks = np.append(qks, qk)
        qms = np.append(qms, qm)

        # Look at the breakpoint. If it is discontinous in
        # value put add it here (i.e. so we have above followed
        # by below for the next step). Othersie we can skip it
        # (and it gets adder in the next iteration). But we need
        # to hadle k = 0 carefully (always stick in the origin)
        cond = (k == nbps + 1) if outwards else (k == 0)
        if cond:
            # Add the value boundary
            rs = model.density_poly.breakpoints[k]
            ds = model.r_earth - rs
            dens = model.density(rs)
            vp = model.vp(rs)
            vs = model.vs(rs)
            qk = model.qkappa(rs)
            qm = model.qshear(rs)
            radii = np.append(radii, rs)
            depths = np.append(depths, ds)
            densities = np.append(densities, dens)
            vps = np.append(vps, vp)
            vss = np.append(vss, vs)
            qks = np.append(qks, qk)
            qms = np.append(qms, qm)
        elif model.density(model.density_poly.breakpoints[k]) != model.density(
            model.density_poly.breakpoints[k], break_down=True
        ):
            # Add the value above the inner boundary of this layer
            rs = model.density_poly.breakpoints[k]
            ds = model.r_earth - rs
            dens = model.density(rs, break_down=outwards)
            vp = model.vp(rs, break_down=outwards)
            vs = model.vs(rs, break_down=outwards)
            qk = model.qkappa(rs, break_down=outwards)
            qm = model.qshear(rs, break_down=outwards)
            radii = np.append(radii, rs)
            depths = np.append(depths, ds)
            densities = np.append(densities, dens)
            vps = np.append(vps, vp)
            vss = np.append(vss, vs)
            qks = np.append(qks, qk)
            qms = np.append(qms, qm)

    result = np.rec.fromarrays(
        [depths, radii, densities, vps, vss, qks, qms],
        names="depth, radius, density, vp, vs, qkappa, qshear",
    )
    return result


def _setup_mass_poly(density_poly: PP, breakpoints: np.ndarray) -> PP:
    # setup polynomials for mass. This is 4*pi*\int rho(r)*r^2 dr
    r2_params = np.tile(np.array([0.0, 0.0, 1000.0**3]), (breakpoints.size - 1, 1))
    r2_poly = PP(r2_params, breakpoints)
    mass_poly = density_poly.mult(r2_poly)
    mass_poly = mass_poly.antiderivative()
    #  integrating this gives mass:
    mass_poly.coeffs = mass_poly.coeffs * 4.0 * np.pi
    return mass_poly


def _setup_moi_poly(density_poly: PP, breakpoints: np.ndarray) -> PP:
    # setup polynomials for MOI. This is 2/3*4*pi*\int rho(r)*r^4 dr
    r4_params = np.tile(
        np.array([0.0, 0.0, 0.0, 0.0, 1000.0**5]), (breakpoints.size - 1, 1)
    )
    r4_poly = PP(r4_params, breakpoints)
    moi_poly = density_poly.mult(r4_poly)
    moi_poly = moi_poly.antiderivative()
    #   integrating this gives MOI:
    moi_poly.coeffs = moi_poly.coeffs * 4.0 * (2 / 3) * np.pi
    return moi_poly


def _setup_gravity_poly(density_poly: PP, breakpoints: np.ndarray) -> PP:
    r2_params = np.tile(np.array([0.0, 0.0, 1000.0**3]), (breakpoints.size - 1, 1))
    r2_poly = PP(r2_params, breakpoints)
    # Setup polynomial for gravity
    gravity_poly = density_poly.mult(r2_poly)
    # evaluate this to get int(rho.r^2 dr)
    gravity_poly = gravity_poly.integrating_poly()
    # constants outside integral
    gravity_poly.coeffs = gravity_poly.coeffs * 4.0 * np.pi * G
    over_r_sq_poly = PP(
        np.zeros((breakpoints.size - 1, 1)),
        breakpoints,
        np.zeros((breakpoints.size - 1, 3)),
    )
    over_r_sq_poly.negative_coeffs[:, 2] = 1.0 / 1000.0**2
    gravity_poly = gravity_poly.mult(over_r_sq_poly)  # Mult by 1/r^2
    gravity_poly = gravity_poly  # Evaluate to get gravity at r
    return gravity_poly


def _setup_pressure_poly(
    gravity_poly: PP, density_poly: PP
) -> PP:  # breakpoints not needed?
    # Setup polynomial for pressure:
    # integrate from r to r_earth to get pressure
    pressure_poly = gravity_poly.mult(density_poly)
    pressure_poly = pressure_poly.antiderivative()  # THIS OVERRIDES THE PREVIOUS LINE?
    # Pressure units (/1E9) and density units (*1000.0)
    pressure_poly.coeffs *= 1000.0 / 1.0e9
    pressure_poly.negative_coeffs *= 1000.0 / 1.0e9
    return pressure_poly


def calculate_vs(
    vs_poly: PP,
    r: float,
    t: float = 1,
    qm_poly: PP | None = None,
    break_down: bool = False,
) -> float | np.ndarray:
    """
    Evaluate s-wave velocity (in km/s) at radius r (in km).

    Optionally corrected for period (t), default is 1 s.
    """
    vs = vs_poly(r, break_down=break_down)
    if t != 1:
        if qm_poly is None:
            raise ValueError("qm_poly must be provided for attenuation correction")
        qm = qm_poly(r, break_down=break_down)
        vs = vs * (1.0 - ((np.log(t) / np.pi) * np.reciprocal(qm)))

    return vs


def calculate_vp(
    vp_poly: PP,
    r: float,
    t: float = 1,
    qk_poly: PP | None = None,
    qm_poly: PP | None = None,
    vs_poly: PP | None = None,
    break_down: bool = False,
) -> float | np.ndarray:
    """
    Evaluate p-wave velocity (in km/s) at radius r (in km).

    Optionally corrected for period (t), default is 1 s.
    """
    vp = vp_poly(r, break_down=break_down)
    if t != 1:
        if qk_poly is None or qm_poly is None or vs_poly is None:
            raise ValueError(
                "qk_poly, qm_poly and vs_poly must be provided for attenuation correction"
            )
        qm = qm_poly(r, break_down=break_down)
        qk = qk_poly(r, break_down=break_down)
        vs = vs_poly(r, break_down=break_down)
        e = (4 / 3) * ((vs / vp) ** 2)
        vp = vp * (
            1.0
            - (
                (np.log(t) / np.pi)
                * (((1.0 - e) * np.reciprocal(qk)) - e * np.reciprocal(qm))
            )
        )
    return vp


def calculate_bulk_modulus(
    vp_poly: PP, vs_poly: PP, density_poly: PP, r: float
) -> float:
    """
    Evaluate bulk modulus (in GPa) at radius r (in km)
    """
    vp = vp_poly(r) * 1000.0  # m/s
    mu = calculate_shear_modulus(vs_poly, density_poly, r)
    density = density_poly(r)
    return ((vp**2 * density) / 1e9) - mu


def calculate_shear_modulus(vs_poly: PP, density_poly: PP, r: float) -> float:
    """
    Evaluate shear modulus (in GPa) at radius r (in km)
    """
    vs = vs_poly(r) * 1000.0  # m/s
    density = density_poly(r)
    return (vs**2 * density) / 1.0e9


def calculate_mass(mass_poly: PP, r: np.ndarray, r_inner=0.0):
    """
    Evaluate mass inside radius r (in km)
    """
    if np.ndim(r) == 0:
        m = mass_poly.integrate(r_inner, r)
    else:
        m = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                m[i] = 0
            else:
                m[i] = mass_poly.integrate(r_inner, r[i])
    return m


def calculate_moi(moi_poly: PP, mass_poly: PP, r: np.ndarray, r_inner=0.0):
    """
    Evaluate moment of inertia inside radius r (in km)

    Return a tuple of moment of inertia (in kg m^2) and
    the moment of inertia factor (I/MR**2, dimensionless)
    which is 0.4 for a uniform density body, and decreases
    as the core becomes more dense than the crust/mantle.

    """
    if np.ndim(r) == 0:
        moi = moi_poly.integrate(r_inner, r)
    else:
        moi = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                moi[i] = 0
            else:
                moi[i] = moi_poly.integrate(r_inner, r[i])

    r_in_m = r * 1000
    m = calculate_mass(mass_poly, r, r_inner=r_inner)
    moif = moi / (m * (r_in_m**2))
    return moi, moif


def calculate_gravity(gravity_poly: PP, r: np.ndarray):
    if np.ndim(r) == 0:
        if r == 0.0:
            return 0.0
        g = gravity_poly(r)
    else:
        g = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                g[i] = 0
            else:
                g[i] = gravity_poly(r[i])
    return g


def calculate_grav_potential(mass_poly: PP, r: np.ndarray):
    """
    Evaluate the gravitational potential at radius r in J/kg
    """
    if np.ndim(r) == 0:
        phi = -1 * mass_poly.integrate(0.0, r) / (r * 1000) * G
    else:
        phi = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                phi[i] = 0
            else:
                phi[i] = -1 * mass_poly.integrate(0.0, r[i]) / (r[i] * 1000) * G
    return phi


def calculate_pressure(pressure_poly: PP, r):
    """
    Evaluate pressure (in GPa) at radius r (in km)
    """
    if np.ndim(r) == 0:
        p = pressure_poly.integrate(r, R_EARTH)
    else:
        p = np.zeros_like(r)
        for i in range(r.size):
            p[i] = pressure_poly.integrate(r[i], R_EARTH)
    return p
