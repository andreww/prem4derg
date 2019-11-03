#!/usr/bin/env python
# coding=utf8

"""
Support for PREM-like 1D Earth models

"""

import numpy as np
import scipy.integrate as spint

import peice_poly as pp


# Default parameters for isotropic PREM
_r_earth = 6371.0
_bps = np.array([0.0, 1221.5, 3480.0, 3630.0, 5600.0, 5701.0, 5771.0,
                 5971.0, 6151.0, 6291.0, 6346.6, 6356.0, 6368.0, 6371.0])
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
                            [2.6000,  0.0000,  0.0000,  0.0000],
                            [1.0200,  0.0000,  0.0000,  0.0000]])
_density_params[:,0] = _density_params[:,0] * 1000.0
_density_params[:,1] = (_density_params[:,1] * 1000.0) / _r_earth 
_density_params[:,2] = (_density_params[:,2] * 1000.0) / (_r_earth**2)
_density_params[:,3] = (_density_params[:,3] * 1000.0) / (_r_earth**3)
_vp_params = np.array([[11.2622,   0.0000, -6.3640,   0.0000],
                      [11.0487,  -4.0362,  4.8023, -13.5732],
                      [15.3891,  -5.3181,  5.5242,  -2.5514],
                      [24.9520, -40.4673, 51.4832, -26.6419],
                      [29.2766, -23.6027,  5.5242,  -2.5514],
                      [19.0957,  -9.8672,  0.0000,   0.0000],
                      [39.7027, -32.6166,  0.0000,   0.0000],
                      [20.3926, -12.2569,  0.0000,   0.0000],
                      [ 4.1875,   3.9382,  0.0000,   0.0000],
                      [ 4.1875,   3.9382,  0.0000,   0.0000],
                      [ 6.8000,   0.0000,  0.0000,   0.0000],
                      [ 5.8000,   0.0000,  0.0000,   0.0000],
                      [ 1.4500,   0.0000,  0.0000,   0.0000]])
_vs_params = np.array([[ 3.6678,   0.0000,  -4.4475,  0.0000],
                      [ 0.0000,   0.0000,   0.0000,  0.0000],
                      [ 6.9254,   1.4672,  -2.0834,  0.9783],
                      [11.1671, -13.7818,  17.4575, -9.2777],
                      [22.3459, -17.2473,  -2.0834,  0.9783],
                      [ 9.9839,  -4.9324,   0.0000,  0.0000],
                      [22.3512, -18.5856,   0.0000,  0.0000],
                      [ 8.9496,  -4.4597,   0.0000,  0.0000],
                      [ 2.1519,   2.3481,   0.0000,  0.0000],
                      [ 2.1519,   2.3481,   0.0000,  0.0000],
                      [ 3.9000,   0.0000,   0.0000,  0.0000],
                      [ 3.2000,   0.0000,   0.0000,  0.0000],
                      [ 0.0000,   0.0000,   0.0000,  0.0000]])
# Turn range of polynomials from 0 - 1 to 0 - r_earth
_vp_params[:,1] = _vp_params[:,1] / _r_earth 
_vp_params[:,2] = _vp_params[:,2] / (_r_earth**2)
_vp_params[:,3] = _vp_params[:,3] / (_r_earth**3)
# Turn range of polynomials from 0 - 1 to 0 - r_earth
_vs_params[:,1] = _vs_params[:,1] / _r_earth 
_vs_params[:,2] = _vs_params[:,2] / (_r_earth**2)
_vs_params[:,3] = _vs_params[:,3] / (_r_earth**3)
_q_kappa_params = np.array([1327.7, 57823.0, 57823.0, 57823.0, 57823.0,
                           57823.0, 57823.0, 57823.0, 57823.0, 57823.0,
                           57823.0, 57823.0, 57823.0])
_q_mu_params = np.array([84.6, np.inf, 312.0, 312.0, 312.0, 143.0, 143.0,
                        143.0, 80.0, 600.0, 600.0, 600.0, np.inf])

class Prem(object):
    
    def __init__(self, breakpoints=_bps, density_params=_density_params,
                 vp_params=_vp_params, vs_params=_vs_params, 
                 q_mu_params=_q_mu_params, q_kappa_params=_q_kappa_params,
                 r_earth=_r_earth):
        
        self.r_earth = r_earth
        
        self.density_poly = pp.PeicewisePolynomial(density_params, 
                                           breakpoints)
        self.vp_poly = pp.PeicewisePolynomial(vp_params, breakpoints)
        self.vs_poly = pp.PeicewisePolynomial(vs_params, breakpoints)
        self.qk_poly = pp.PeicewisePolynomial(q_kappa_params, breakpoints)
        self.qm_poly = pp.PeicewisePolynomial(q_mu_params, breakpoints)
        
        # setup polynomials for mass. This is 4*pi*\int rho(r)*r^2 dr
        r2_params = np.tile(np.array([0.0, 0.0, 1000.0**3]),
                            (breakpoints.size-1,1))
        r2_poly = pp.PeicewisePolynomial(r2_params, breakpoints)
        self.mass_poly = self.density_poly.mult(r2_poly)
        self.mass_poly = self.mass_poly.antiderivative()
        # integrating this gives mass:
        self.mass_poly.coeffs = self.mass_poly.coeffs * 4.0 * np.pi
        
        # setup polynomials for MOI. This is 2/3*4*pi*\int rho(r)*r^4 dr
        r4_params = np.tile(np.array([0.0,  0.0, 0.0, 0.0, 1000.0**5]),
                            (breakpoints.size-1,1))
        r4_poly = pp.PeicewisePolynomial(r4_params, breakpoints)
        self.moi_poly = self.density_poly.mult(r4_poly) 
        self.moi_poly = self.moi_poly.antiderivative()
        # integrating this gives MOI:
        self.moi_poly.coeffs = self.moi_poly.coeffs * 4.0 * (2/3) * np.pi  
        
    def density(self, r):
        """
        Evaluate density in kg/m**3 at radii r (in km)
        """
        return self.density_poly(r)

    def vs(self, r, t=1):
        """
        Evaluate s-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        vs = self.vs_poly(r)
        if t != 1:
            qm = self.qm_poly(r)
            vs = vs * (1.0 - ((np.log(t)/np.pi)*np.reciprocal(qm)))
        
        return vs
    
    def vp(self, r, t=1):
        """
        Evaluate p-wave velocity (in km/s) at radius r (in km).

        Optionally corrected for period (t), default is 1 s.
        """
        vp = self.vp_poly(r)
        if t != 1:
            qm = self.qm_poly(r)
            qk = self.qk_poly(r)
            vs = self.vs_poly(r)
            e = (4/3)*((vs/vp)**2)
            vp = vp * (1.0 - ((np.log(t)/np.pi)*(((1.0-e)*np.reciprocal(qk)) -
                                                 e*np.reciprocal(qm))))
        return vp

    def mass(self, r, r_inner=0.0):
        """
        Evaluate mass inside radius r (in km)
        """
        if np.ndim(r) == 0:
            m = self.mass_poly.integrate(r_inner,r)
        else:
            m = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    m[i] = 0
                else:
                    m[i] = self.mass_poly.integrate(r_inner,r[i])
        return m
    
    def moment_or_inertia(self, r, r_inner=0.0):
        """
        Evaluate moment of inertia inside radius r (in km)
        
        Return a tuple of moment of inertia (in kg m^2) and
        the moment of inertia factor (I/MR**2, dimensionless)
        which is 0.4 for a uniform density body, and decreases
        as the core becomes more dense than the crust/mantle.

        """
        if np.ndim(r) == 0:
            moi = self.moi_poly.integrate(r_inner,r)
        else:
            moi = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    moi[i] = 0
                else:
                    moi[i] = self.moi_poly.integrate(r_inner,r[i])
                    
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
            g = self.mass_poly.integrate(0.0,r)/((r*1000)**2)*G
        else:
            g = np.zeros_like(r)
            for i in range(r.size):
                if r[i] == 0:
                    g[i] = 0
                else:
                    g[i] = self.mass_poly.integrate(0.0, r[i]) / \
                                                      ((r[i]*1000)**2)*G 
        return g
    
    def pressure(self, r):
        """
        Evaluate pressure (in GPa) at radius r (in km)
        """
        # NB: this is done numerically, because otherise
        # I need to work out how to express g as a polynomial,
        # multiply this by rho, then do the integral inwards.
        # all a bit of a faff!
        if np.ndim(r) == 0:
            # 10 km grid spacing
            rs = np.arange(r, self.r_earth, 1.0)
            g = self.gravity(rs)
            rho = self.density(rs)
            ps = spint.cumtrapz((-g*rho)[::-1],rs[::-1]*1000.0, initial=0)
            pressure = ps[-1]/1E9
        else:
            # Assume I have been fed something I can integrate
            g = self.gravity(r)
            rho = self.density(r)
            pressure = spint.cumtrapz((-g*rho)[::-1],
                                      r[::-1]*1000.0, initial=0)
            pressure = pressure[::-1]/1E9
        return pressure
