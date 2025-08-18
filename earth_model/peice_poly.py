#!/usr/bin/env python
# coding=utf8
"""
Peicewise polynomials like PREM

"""
import numpy as np


class PeicewisePolynomial(object):
    """
    Peicewise Polynomials a different way

    The SciPy PPoly class defines a function from
    polynomials with coefficents c and breakpoints x
    evaluated at a point xp thus:

       S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    This is not helpful for PREM, so we create a new class defining
    the function:

       S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    Note some important differences between this and PPoly!
    """

    def __init__(self, c, x):
        assert len(x.shape) == 1, "breakpoints must be 1D"
        self.breakpoints = x
        if len(c.shape) == 1:
            c = np.expand_dims(c, axis=1)
            c = np.append(c, np.zeros_like(c), axis=1)
        assert len(c.shape) == 2, "breakpoints must be 2D"
        self.coeffs = c

    def __call__(self, xp, break_down=False):
        if np.ndim(xp) == 0:
            value = self._evaluate_at_point(xp, break_down)
        else:
            value = np.zeros_like(xp)
            for i in range(xp.size):
                value[i] = self._evaluate_at_point(xp[i], break_down)
        return value

    def _evaluate_at_point(self, x, break_down=False):
        """
        Evaluate piecewise polynomial at point x
        """
        coef = self._get_coefs(x, break_down)
        value = self._evaluate_polynomial(x, coef)
        return value

    def _evaluate_polynomial(self, x, coef):
        value = 0
        for i, c in enumerate(coef):
            value = value + c * x**i
        return value

    def _get_coefs(self, x, break_down=False):
        """
        Return coefs at x

        If x falls on a breakpoint, we take the coeffecents from
        'above' the breakpoint. Unless break_down is True, in which
        case we take the coeffecents from 'below'
        """
        if x == self.breakpoints[-1]:
            # We use the last coefficents for the outside point
            return self.coeffs[-1, :]
        if break_down:
            for i in range(self.breakpoints.size):
                if ((x > self.breakpoints[i])
                   and (x <= self.breakpoints[i+1])):
                    return self.coeffs[i, :]
        else:
            for i in range(self.breakpoints.size):
                if ((x >= self.breakpoints[i])
                   and (x < self.breakpoints[i+1])):
                    return self.coeffs[i, :]

        return None

    def derivative(self):
        deriv_breakpoints = self.breakpoints
        deriv_coeffs = np.zeros((self.coeffs.shape[0],
                                 self.coeffs.shape[1]-1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                if i == 0:
                    continue  # Throw away term for x**0
                deriv_coeffs[seg, i-1] = self.coeffs[seg, i]*i

        deriv = PeicewisePolynomial(deriv_coeffs, deriv_breakpoints)
        return deriv

    def antiderivative(self):
        antideriv_breakpoints = self.breakpoints
        antideriv_coeffs = np.zeros((self.coeffs.shape[0],
                                     self.coeffs.shape[1]+1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                antideriv_coeffs[seg, i+1] = self.coeffs[seg, i]/(i+1)

        antideriv = PeicewisePolynomial(antideriv_coeffs,
                                        antideriv_breakpoints)
        return antideriv

    def integrate(self, a, b):
        # antiderivative = self.antiderivative()
        integral = 0
        lower_bound = a
        for bpi, bp in enumerate(self.breakpoints):
            if bp > lower_bound:
                if self.breakpoints[bpi] >= b:
                    # Just the one segment left - add it and end
                    integral = integral + (self(b, break_down=True) -
                                           self(lower_bound))
                    # print(integral, lower_bound, b, 'done')
                    break
                else:
                    # segment from lower bound to bp
                    # add it, increment lower_bound and contiue
                    integral = integral + (self(bp, break_down=True) -
                                           self(lower_bound))
                    # print(integral, lower_bound, bp)
                    lower_bound = bp

        return integral
    
    def integrating_poly(self):
        """
        Returns a piecewise polynomial that represents the definite
        integral self between 0 and the evaluation point.
        """
        antiderivative = self.antiderivative()
        ip_coeffs = np.zeros_like(antiderivative.coeffs)
        # Inside each segment, the integral between 0 and x
        # is the integral between the lower bound of that 
        # segment and the upper bound, plus the integral 
        # for all other segments. These are all constants
        # so can be added to the constent term in this 
        # segment's antiderivative (we dont need the last breakpoint)
        for bpi, bp in enumerate(antiderivative.breakpoints[0:-1]):
            ip_coeffs[bpi, :] = antiderivative.coeffs[bpi, :]
            # Subtract antiderivate on inner boundary
            ip_coeffs[bpi, 0] = ip_coeffs[bpi, 0] - antiderivative(bp)
            # add all the other segments
            if bpi > 0:
                ip_coeffs[bpi, 0] = ip_coeffs[bpi, 0] + antiderivative.integrate(0, bp)
 
        return PeicewisePolynomial(ip_coeffs, antiderivative.breakpoints)

    def mult(self, other):
        # FIXME - for this approach brakepoints need to be same place too
        assert self.coeffs.shape[0] == other.coeffs.shape[0], \
                                     'different number of breakpoints'
        mult_breakpoints = self.breakpoints
        mult_coefs = np.zeros((self.coeffs.shape[0],
                               self.coeffs.shape[1]+other.coeffs.shape[1]))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                for j in range(other.coeffs.shape[1]):
                    mult_coefs[seg, i+j] = mult_coefs[seg, i+j] + \
                                 self.coeffs[seg, i] * other.coeffs[seg, j]

        mult_poly = PeicewisePolynomial(mult_coefs, mult_breakpoints)
        return mult_poly
    
    def div_xsq(self):
        """
        The polynomial is multiplied by x^-2
        """
        # Ideally we should implement polynomial division, but
        # this is all I need for now
        assert np.allclose(self.coeffs[:, 0], 0.0)
        assert np.allclose(self.coeffs[:, 1], 0.0) 
        div_breakpoints = self.breakpoints
        # degree of self - degree of other must be positive
        div_coeffs = np.zeros((self.coeffs.shape[0], self.coeffs.shape[1]-2))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                div_coeffs[seg, i-2] = self.coeffs[seg, i] 
        div_poly = PeicewisePolynomial(div_coeffs, div_breakpoints)
        return div_poly