#!/usr/bin/env python
# coding=utf8
"""
Support for PREM-like 1D Earth models

"""

import numpy as np
import scipy.integrate as spint  # type: ignore

from . import peice_poly as pp


# Default parameters for isotropic PREM
_r_earth = 6371.0
_bps = np.array([0.0, 1221.5, 3480.0, 3630.0, 5600.0, 5701.0, 5771.0,
                 5971.0, 6151.0, 6291.0, 6346.6, 6356.0, 6371.0])
_density_params = np.array([[13.0885,  0.0000, -8.8381,  0.0000],
                            [12.5815, -1.2638, -3.6426, -5.5281],
                            [7.9565, -6.4761,  5.5283, -3.0807],
                            [7.9565, -6.4761,  5.5283, -3.0807],
                            [7.9565, -6.4761,  5.5283, -3.0807],
                            [5.3197, -1.4836,  0.0000,  0.0000],
                            [11.2494, -8.0298,  0.0000,  0.0000],
                            [7.1089, -3.8045,  0.00002,  0.0000],
                            [2.6910,  0.6924,  0.0000,  0.0000],
                            [2.6910,  0.6924,  0.0000,  0.0000],
                            [2.9000,  0.0000,  0.0000,  0.0000],
                            [2.6000,  0.0000,  0.0000,  0.0000]])
_density_params[:, 0] = _density_params[:, 0] * 1000.0
_density_params[:, 1] = (_density_params[:, 1] * 1000.0) / _r_earth
_density_params[:, 2] = (_density_params[:, 2] * 1000.0) / (_r_earth**2)
_density_params[:, 3] = (_density_params[:, 3] * 1000.0) / (_r_earth**3)
_vp_params = np.array([[11.2622,   0.0000, -6.3640,   0.0000],
                      [11.0487,  -4.0362,  4.8023, -13.5732],
                      [15.3891,  -5.3181,  5.5242,  -2.5514],
                      [24.9520, -40.4673, 51.4832, -26.6419],
                      [29.2766, -23.6027,  5.5242,  -2.5514],
                      [19.0957,  -9.8672,  0.0000,   0.0000],
                      [39.7027, -32.6166,  0.0000,   0.0000],
                      [20.3926, -12.2569,  0.0000,   0.0000],
                      [4.1875,   3.9382,  0.0000,   0.0000],
                      [4.1875,   3.9382,  0.0000,   0.0000],
                      [6.8000,   0.0000,  0.0000,   0.0000],
                      [5.8000,   0.0000,  0.0000,   0.0000]])
_vs_params = np.array([[3.6678,   0.0000,  -4.4475,  0.0000],
                      [0.0000,   0.0000,   0.0000,  0.0000],
                      [6.9254,   1.4672,  -2.0834,  0.9783],
                      [11.1671, -13.7818,  17.4575, -9.2777],
                      [22.3459, -17.2473,  -2.0834,  0.9783],
                      [9.9839,  -4.9324,   0.0000,  0.0000],
                      [22.3512, -18.5856,   0.0000,  0.0000],
                      [8.9496,  -4.4597,   0.0000,  0.0000],
                      [2.1519,   2.3481,   0.0000,  0.0000],
                      [2.1519,   2.3481,   0.0000,  0.0000],
                      [3.9000,   0.0000,   0.0000,  0.0000],
                      [3.2000,   0.0000,   0.0000,  0.0000]])
# Turn range of polynomials from 0 - 1 to 0 - r_earth
_vp_params[:, 1] = _vp_params[:, 1] / _r_earth
_vp_params[:, 2] = _vp_params[:, 2] / (_r_earth**2)
_vp_params[:, 3] = _vp_params[:, 3] / (_r_earth**3)
# Turn range of polynomials from 0 - 1 to 0 - r_earth
_vs_params[:, 1] = _vs_params[:, 1] / _r_earth
_vs_params[:, 2] = _vs_params[:, 2] / (_r_earth**2)
_vs_params[:, 3] = _vs_params[:, 3] / (_r_earth**3)
_q_kappa_params = np.array([1327.7, 57823.0, 57823.0, 57823.0, 57823.0,
                           57823.0, 57823.0, 57823.0, 57823.0, 57823.0,
                           57823.0, 57823.0])
_q_mu_params = np.array([84.6, np.inf, 312.0, 312.0, 312.0, 143.0, 143.0,
                        143.0, 80.0, 600.0, 600.0, 600.0])


class Prem(object):

    def __init__(self, breakpoints=_bps, density_params=_density_params,
                 vp_params=_vp_params, vs_params=_vs_params,
                 q_mu_params=_q_mu_params, q_kappa_params=_q_kappa_params,
                 r_earth=_r_earth):

        self.r_earth = r_earth

        self.density_poly = pp.PeicewisePolynomial(density_params, breakpoints)
        self.vp_poly = pp.PeicewisePolynomial(vp_params, breakpoints)
        self.vs_poly = pp.PeicewisePolynomial(vs_params, breakpoints)
        self.qk_poly = pp.PeicewisePolynomial(q_kappa_params, breakpoints)
        self.qm_poly = pp.PeicewisePolynomial(q_mu_params, breakpoints)

        # setup polynomials for mass. This is 4*pi*\int rho(r)*r^2 dr
        r2_params = np.tile(np.array([0.0, 0.0, 1000.0**3]),
                            (breakpoints.size - 1, 1))
        r2_poly = pp.PeicewisePolynomial(r2_params, breakpoints)
        self.mass_poly = self.density_poly.mult(r2_poly)
        self.mass_poly = self.mass_poly.antiderivative()
        #  integrating this gives mass:
        self.mass_poly.coeffs = self.mass_poly.coeffs * 4.0 * np.pi

        # setup polynomials for MOI. This is 2/3*4*pi*\int rho(r)*r^4 dr
        r4_params = np.tile(np.array([0.0,  0.0, 0.0, 0.0, 1000.0**5]),
                            (breakpoints.size - 1, 1))
        r4_poly = pp.PeicewisePolynomial(r4_params, breakpoints)
        self.moi_poly = self.density_poly.mult(r4_poly)
        self.moi_poly = self.moi_poly.antiderivative()
        #   integrating this gives MOI:
        self.moi_poly.coeffs = self.moi_poly.coeffs * 4.0 * (2/3) * np.pi

    def density(self, r, break_down=False):
        """
        Evaluate density in kg/m**3 at radii r (in km)
        """
        return self.density_poly(r, break_down=break_down)

    def vs(self, r, t=1, break_down=False):
        """
        Evaluate s-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        vs = self.vs_poly(r, break_down=break_down)
        if t != 1:
            qm = self.qm_poly(r, break_down=break_down)
            vs = vs * (1.0 - ((np.log(t)/np.pi)*np.reciprocal(qm)))

        return vs

    def vp(self, r, t=1, break_down=False):
        """
        Evaluate p-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        vp = self.vp_poly(r, break_down=break_down)
        if t != 1:
            qm = self.qm_poly(r, break_down=break_down)
            qk = self.qk_poly(r, break_down=break_down)
            vs = self.vs_poly(r, break_down=break_down)
            e = (4/3)*((vs/vp)**2)
            vp = vp * (1.0 - ((np.log(t)/np.pi)*(((1.0-e)*np.reciprocal(qk)) -
                                                 e*np.reciprocal(qm))))
        return vp

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
        vp = self.vp_poly(r) * 1000.0  # m/s
        mu = self.shear_modulus(r)
        density = self.density_poly(r)
        return ((vp**2 * density) / 1e9) - mu

    def shear_modulus(self, r):
        """
        Evaluate shear modulus (in GPa) at radius r (in km)
        """
        vs = self.vs_poly(r) * 1000.0  # m/s
        density = self.density_poly(r)
        return (vs**2 * density) / 1.0e9

    def mass(self, r, r_inner=0.0):
        """
        Evaluate mass inside radius r (in km)
        """
        if np.ndim(r) == 0:
            m = self.mass_poly.integrate(r_inner, r)
        else:
            m = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    m[i] = 0
                else:
                    m[i] = self.mass_poly.integrate(r_inner, r[i])
        return m

    def moment_of_inertia(self, r, r_inner=0.0):
        """
        Evaluate moment of inertia inside radius r (in km)

        Return a tuple of moment of inertia (in kg m^2) and
        the moment of inertia factor (I/MR**2, dimensionless)
        which is 0.4 for a uniform density body, and decreases
        as the core becomes more dense than the crust/mantle.

        """
        if np.ndim(r) == 0:
            moi = self.moi_poly.integrate(r_inner, r)
        else:
            moi = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    moi[i] = 0
                else:
                    moi[i] = self.moi_poly.integrate(r_inner, r[i])

        r_in_m = r * 1000
        m = self.mass(r)
        moif = moi / (m*(r_in_m**2))
        return moi, moif

    def gravity(self, r):
        """
        Evaluate acceleration due to gravity at radius r in m/s^2
        """
        G = 6.6743E-11
        if np.ndim(r) == 0:
            if r == 0:
                g = 0
            else:
                g = self.mass_poly.integrate(0.0, r)/((r*1000)**2)*G
        else:
            g = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    g[i] = 0
                else:
                    g[i] = self.mass_poly.integrate(0.0, r[i]) / \
                                                      ((r[i]*1000)**2)*G
        return g

    def grav_potential(self, r):
        """
        Evaluate the gravitational potential at radius r in J/kg
        """
        G = 6.6743E-11
        if np.ndim(r) == 0:
            phi = -1 * self.mass_poly.integrate(0.0, r)/(r*1000)*G
        else:
            phi = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    phi[i] = 0
                else:
                    phi[i] = -1 * self.mass_poly.integrate(0.0, r[i]) / \
                                                          (r[i]*1000)*G
        return phi

    def pressure(self, r):
        """
        Evaluate pressure (in GPa) at radius r (in km)
        """
        # NB: this is done numerically, because otherise
        # I need to work out how to express g as a polynomial,
        # multiply this by rho, then do the integral inwards.
        # all a bit of a faff!
        if np.ndim(r) == 0:
            # 10 km grid spacing
            rs = np.arange(r, self.r_earth, 1.0)
            g = self.gravity(rs)
            rho = self.density(rs)
            ps = spint.cumulative_trapezoid((-g*rho)[::-1], rs[::-1]*1000.0, initial=0)
            pressure = ps[-1]/1E9
        else:
            # Assume I have been fed something I can integrate
            g = self.gravity(r)
            rho = self.density(r)
            pressure = spint.cumulative_trapezoid((-g*rho)[::-1],
                                      r[::-1]*1000.0, initial=0)
            pressure = pressure[::-1]/1E9
        return pressure

    def tabulate_model_inwards(self, min_step):
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

        and is ordered such that element 0 is at the surface and the last
        element (element -1) is at the center of the planet.
        """
        # Keep the data as we get it
        radii = np.array([])
        depths = np.array([])
        densities = np.array([])
        vps = np.array([])
        vss = np.array([])
        qks = np.array([])
        qms = np.array([])

        nbps = len(self.density_poly.breakpoints) - 1
        for i in range(nbps):
            j = nbps - i
            k = j - 1
            rs = np.arange(self.density_poly.breakpoints[j],
                           self.density_poly.breakpoints[k], -min_step)
            ds = self.r_earth - rs
            dens = self.density(rs, break_down=True)  # As we go inwards
            vp = self.vp(rs, break_down=True)  # As we go inwards
            vs = self.vs(rs, break_down=True)  # As we go inwards
            qk = self.qkappa(rs, break_down=True)  # As we go inwards
            qm = self.qshear(rs, break_down=True)  # As we go inwards
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
            if k == 0:
                # Add the value at r=0
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs)
                vp = self.vp(rs)
                vs = self.vs(rs)
                qk = self.qkappa(rs)
                qm = self.qshear(rs)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)
            elif (self.density(self.density_poly.breakpoints[k]) !=
                  self.density(self.density_poly.breakpoints[k],
                               break_down=True)):
                # Add the value above the inner boundary of this layer
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs)
                vp = self.vp(rs)
                vs = self.vs(rs)
                qk = self.qkappa(rs)
                qm = self.qshear(rs)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)

        result = np.core.records.fromarrays(
            [depths, radii, densities, vps, vss, qks, qms],
            names='depth, radius, density, vp, vs, qkappa, qshear'
            )
        return result

    def tabulate_model_outwards(self, min_step):
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

        and is ordered such that element 0 is at the center of the planet
        and the last element (element -1) is at the surface.
        """
        # Keep the data as we get it
        radii = np.array([])
        depths = np.array([])
        densities = np.array([])
        vps = np.array([])
        vss = np.array([])
        qks = np.array([])
        qms = np.array([])

        nbps = len(self.density_poly.breakpoints) - 1
        for i in range(nbps):
            j = i
            k = j + 1
            rs = np.arange(self.density_poly.breakpoints[j],
                           self.density_poly.breakpoints[k], min_step)
            ds = self.r_earth - rs
            dens = self.density(rs)
            vp = self.vp(rs)
            vs = self.vs(rs)
            qk = self.qkappa(rs)
            qm = self.qshear(rs)
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
            if k == nbps + 1:
                # Add the value surface
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs)
                vp = self.vp(rs)
                vs = self.vs(rs)
                qk = self.qkappa(rs)
                qm = self.qshear(rs)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)
            elif (self.density(self.density_poly.breakpoints[k]) !=
                  self.density(self.density_poly.breakpoints[k],
                               break_down=True)):
                # Add the value above the inner boundary of this layer
                rs = self.density_poly.breakpoints[k]
                ds = self.r_earth - rs
                dens = self.density(rs, break_down=True)
                vp = self.vp(rs, break_down=True)
                vs = self.vs(rs, break_down=True)
                qk = self.qkappa(rs, break_down=True)
                qm = self.qshear(rs, break_down=True)
                radii = np.append(radii, rs)
                depths = np.append(depths, ds)
                densities = np.append(densities, dens)
                vps = np.append(vps, vp)
                vss = np.append(vss, vs)
                qks = np.append(qks, qk)
                qms = np.append(qms, qm)

        result = np.core.records.fromarrays(
            [depths, radii, densities, vps, vss, qks, qms],
            names='depth, radius, density, vp, vs, qkappa, qshear'
            )
        return result
