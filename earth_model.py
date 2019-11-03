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

class Prem(object):
    
    def __init__(self, breakpoints=_bps, density_params=_density_params,
                 r_earth=_r_earth):
        
        self.r_earth = r_earth
        
        self.density_poly = pp.PeicewisePolynomial(density_params, 
                                           breakpoints)
        
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
