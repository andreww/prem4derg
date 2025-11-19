#!/usr/bin/env python
# coding=utf8
"""
Support for PREM-like 1D Earth models

"""

from dataclasses import dataclass, field
import numpy as np

from .const import R_EARTH
from .peice_poly import PeicewisePolynomial as PP
from .physics import (
    calculate_bulk_modulus,
    calculate_density,
    calculate_grav_potential,
    calculate_gravity,
    calculate_mass,
    calculate_moi,
    calculate_pressure,
    calculate_qkappa,
    calculate_qshear,
    calculate_shear_modulus,
    calculate_vp,
    calculate_vs,
)


@dataclass
class OneDModel:
    breakpoints: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    density_params: np.ndarray = field(
        default_factory=lambda: np.zeros((), dtype=float)
    )
    vp_params: np.ndarray = field(default_factory=lambda: np.zeros((), dtype=float))
    vs_params: np.ndarray = field(default_factory=lambda: np.zeros((), dtype=float))
    q_mu_params: np.ndarray = field(default_factory=lambda: np.zeros((), dtype=float))
    q_kappa_params: np.ndarray = field(
        default_factory=lambda: np.zeros((), dtype=float)
    )
    r_earth: float = R_EARTH

    def __post_init__(self):
        """Initialise piecewise polynomials when params are not 0D."""

        def has_coeffs(arr: np.ndarray) -> bool:
            return arr.ndim != 0

        if has_coeffs(self.density_params):
            self.density_poly = PP(self.density_params, self.breakpoints)
        if has_coeffs(self.vp_params):
            self.vp_poly = PP(self.vp_params, self.breakpoints)
        if has_coeffs(self.vs_params):
            self.vs_poly = PP(self.vs_params, self.breakpoints)
        if has_coeffs(self.q_kappa_params):
            self.qk_poly = PP(self.q_kappa_params, self.breakpoints)
        if has_coeffs(self.q_mu_params):
            self.qm_poly = PP(self.q_mu_params, self.breakpoints)

    def density(self, r, break_down=False):
        """
        Evaluate density in kg/m**3 at radii r (in km)
        """
        dp = self._require_polynomial("density_poly")
        return calculate_density(dp, r, break_down=break_down)

    def vs(self, r, t=1, break_down=False) -> float | np.ndarray:
        """
        Evaluate s-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        vsp = self._require_polynomial("vs_poly")
        qmp = self._require_polynomial("qm_poly")
        return calculate_vs(vsp, r, t=t, qm_poly=qmp, break_down=break_down)

    def vp(self, r, t=1, break_down=False):
        """
        Evaluate p-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        vp_poly = self._require_polynomial("vp_poly")
        qk_poly = self._require_polynomial("qk_poly")
        qm_poly = self._require_polynomial("qm_poly")
        vs_poly = self._require_polynomial("vs_poly")
        return calculate_vp(
            vp_poly,
            r,
            t=t,
            qk_poly=qk_poly,
            qm_poly=qm_poly,
            vs_poly=vs_poly,
            break_down=break_down,
        )

    def qkappa(self, r, break_down=False):
        qk_poly = self._require_polynomial("qk_poly")
        return calculate_qkappa(qk_poly, r, break_down=break_down)

    def qshear(self, r, break_down=False):
        qm_poly = self._require_polynomial("qm_poly")
        return calculate_qshear(qm_poly, r, break_down=break_down)

    def bulk_modulus(self, r):
        """
        Evaluate bulk modulus (in GPa) at radius r (in km)
        """
        vp_poly = self._require_polynomial("vp_poly")
        vs_poly = self._require_polynomial("vs_poly")
        density_poly = self._require_polynomial("density_poly")
        return calculate_bulk_modulus(vp_poly, vs_poly, density_poly, r)

    def shear_modulus(self, r):
        """
        Evaluate shear modulus (in GPa) at radius r (in km)
        """
        vs_poly = self._require_polynomial("vs_poly")
        density_poly = self._require_polynomial("density_poly")
        return calculate_shear_modulus(vs_poly, density_poly, r)

    def mass(self, r, r_inner=0.0):
        """
        Evaluate mass inside radius r (in km)
        """
        density_poly = self._require_polynomial("density_poly")
        return calculate_mass(density_poly, r, r_inner=r_inner)

    def moment_of_inertia(self, r, r_inner=0.0):
        """
        Evaluate moment of inertia inside radius r (in km)

        Return a tuple of moment of inertia (in kg m^2) and
        the moment of inertia factor (I/MR**2, dimensionless)
        which is 0.4 for a uniform density body, and decreases
        as the core becomes more dense than the crust/mantle.

        """
        density_poly = self._require_polynomial("density_poly")
        return calculate_moi(density_poly, r, r_inner=r_inner)

    def gravity(self, r):
        density_poly = self._require_polynomial("density_poly")
        return calculate_gravity(density_poly, r)

    def grav_potential(self, r):
        """
        Evaluate the gravitational potential at radius r in J/kg
        """
        density_poly = self._require_polynomial("density_poly")
        return calculate_grav_potential(density_poly, r)

    def pressure(self, r):
        """
        Evaluate pressure (in GPa) at radius r (in km)
        """
        density_poly = self._require_polynomial("density_poly")
        return calculate_pressure(density_poly, r)

    def _require_polynomial(self, attr: str) -> PP:
        poly = getattr(self, attr)
        if poly is None:
            raise ValueError(f"{attr} polynomial is not defined.")
        return poly


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
