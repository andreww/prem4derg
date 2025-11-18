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

    The module also supports negative powers set by passing the c_neg
    parameter. The c_neg[:, 0] coefficients are for ln terms used for
    integrals.
    """

    def __init__(self, c, x, c_neg=None):
        assert len(x.shape) == 1, "breakpoints must be 1D"
        self.breakpoints = x
        if len(c.shape) == 1:
            c = np.expand_dims(c, axis=1)
            c = np.append(c, np.zeros_like(c), axis=1)
        assert len(c.shape) == 2, "Positive coefficients must be 2D"
        self.coeffs = c
        if c_neg is not None:
            if len(c_neg.shape) == 1:
                c_neg = np.expand_dims(c_neg, axis=1)
                c_neg = np.append(c_neg, np.zeros_like(c), axis=1)
            assert len(c_neg.shape) == 2, "Negative coefficients must be 2D"
            self.negative_coeffs = c_neg
        else:
            self.negative_coeffs = None

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
        coef, neg_coef = self._get_coefs(x, break_down)
        value = 0
        for i, c in enumerate(coef):
            value = value + c * x**i
        if neg_coef is not None:
            for i, c in enumerate(neg_coef):
                if i == 0 and c != 0.0:  # Hum - avoid these...
                    if x == 0.0:
                        raise ValueError  # Cannot do ln(0)
                    else:
                        value = value + c * np.log(np.abs(x))
                elif x == 0.0 and c != 0.0:
                    raise ZeroDivisionError
                elif c != 0.0:
                    value = value + (c / x**i)
                # The c == 0.0 case can be ignored - adding 0.0
        return value

    def _get_coefs(self, x, break_down=False):
        """
        Return coefs at x

        If x falls on a breakpoint, we take the coefficients from
        'above' the breakpoint. Unless break_down is True, in which
        case we take the coefficients from 'below'
        """
        if x == self.breakpoints[-1]:
            # We use the last coefficients for the outside point
            pos_coef = self.coeffs[-1, :]
            if self.negative_coeffs is None:
                neg_coef = None
            else:
                neg_coef = self.negative_coeffs[-1, :]
            return pos_coef, neg_coef
        if break_down:
            for i in range(self.breakpoints.size):
                if (x > self.breakpoints[i]) and (x <= self.breakpoints[i + 1]):
                    pos_coef = self.coeffs[i, :]
                    if self.negative_coeffs is None:
                        neg_coef = None
                    else:
                        neg_coef = self.negative_coeffs[i, :]
                    return pos_coef, neg_coef
        else:
            for i in range(self.breakpoints.size):
                if (x >= self.breakpoints[i]) and (x < self.breakpoints[i + 1]):
                    pos_coef = self.coeffs[i, :]
                    if self.negative_coeffs is None:
                        neg_coef = None
                    else:
                        neg_coef = self.negative_coeffs[i, :]
                    return pos_coef, neg_coef
        return None, None

    def derivative(self):
        deriv_breakpoints = self.breakpoints
        deriv_coeffs = np.zeros((self.coeffs.shape[0], self.coeffs.shape[1] - 1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                if i == 0:
                    continue  # Throw away term for x**0
                deriv_coeffs[seg, i - 1] = self.coeffs[seg, i] * i

        deriv_neg_coeffs = None
        if self.negative_coeffs is not None:
            deriv_neg_coeffs = np.zeros(
                (self.negative_coeffs.shape[0], self.negative_coeffs.shape[1] + 1)
            )
            for seg in range(self.negative_coeffs.shape[0]):
                for i in range(self.negative_coeffs.shape[1]):
                    if i == 0:
                        # c ln(|x|) term -> c/x
                        deriv_neg_coeffs[seg, 1] = self.negative_coeffs[seg, i]
                    else:
                        deriv_neg_coeffs[seg, i + 1] = (
                            -1 * self.negative_coeffs[seg, i] * i
                        )
        deriv = PeicewisePolynomial(deriv_coeffs, deriv_breakpoints, deriv_neg_coeffs)
        return deriv

    def antiderivative(self):
        antideriv_breakpoints = self.breakpoints
        antideriv_coeffs = np.zeros((self.coeffs.shape[0], self.coeffs.shape[1] + 1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                antideriv_coeffs[seg, i + 1] = self.coeffs[seg, i] / (i + 1)

        antideriv_neg_coeffs = None
        if self.negative_coeffs is not None:
            antideriv_neg_coeffs = np.zeros(
                (self.negative_coeffs.shape[0], self.negative_coeffs.shape[1] - 1)
            )
            for seg in range(self.negative_coeffs.shape[0]):
                for i in range(self.negative_coeffs.shape[1]):
                    if i == 0:
                        assert self.negative_coeffs[seg, i] == 0.0, (
                            "Cannot take antiderivative of ln(|x|) terms"
                        )
                    if i == 1:
                        # c/x term -> c ln(|x|) which we put in i=0. No change
                        # in sign or division
                        antideriv_neg_coeffs[seg, 0] = self.negative_coeffs[seg, i]
                    else:
                        antideriv_neg_coeffs[seg, i - 1] = (
                            -1 * self.negative_coeffs[seg, i] / (i - 1)
                        )
        antideriv = PeicewisePolynomial(
            antideriv_coeffs, antideriv_breakpoints, antideriv_neg_coeffs
        )
        return antideriv

    def integrate(self, a, b):
        # antiderivative = self.antiderivative()
        integral = 0
        lower_bound = a
        for bpi, bp in enumerate(self.breakpoints):
            if bp > lower_bound:
                if self.breakpoints[bpi] >= b:
                    # Just the one segment left - add it and end
                    integral = integral + (self(b, break_down=True) - self(lower_bound))
                    # print(integral, lower_bound, b, 'done')
                    break
                else:
                    # segment from lower bound to bp
                    # add it, increment lower_bound and contiue
                    integral = integral + (
                        self(bp, break_down=True) - self(lower_bound)
                    )
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
        # is the integral between the lower bound of that
        # segment and the upper bound, plus the integral
        # for all other segments. These are all constants
        # so can be added to the constent term in this
        # segment's antiderivative (we dont need the last breakpoint)
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
        assert self.coeffs.shape[0] == other.coeffs.shape[0], (
            "different number of breakpoints"
        )
        mult_breakpoints = self.breakpoints
        mult_coefs = np.zeros(
            (self.coeffs.shape[0], self.coeffs.shape[1] + other.coeffs.shape[1] - 1)
        )
        mult_negative_coefs = None
        if (self.negative_coeffs is not None) and (other.negative_coeffs is not None):
            assert np.all(self.negative_coeffs[:, 0] == 0.0), (
                "Cannot multiply ln(x) terms in self"
            )
            assert np.all(other.negative_coeffs[:, 0] == 0.0), (
                "Cannot multiply ln(x) terms in other"
            )
            mult_negative_coefs = np.zeros(
                (
                    self.negative_coeffs.shape[0],
                    (
                        self.negative_coeffs.shape[1]
                        + other.negative_coeffs.shape[1]
                        - 1
                    ),
                )
            )
        elif self.negative_coeffs is not None:
            assert np.all(self.negative_coeffs[:, 0] == 0.0), (
                "Cannot multiply ln(x) terms in self"
            )
            mult_negative_coefs = np.zeros(
                (self.negative_coeffs.shape[0], self.negative_coeffs.shape[1])
            )
        elif other.negative_coeffs is not None:
            assert np.all(other.negative_coeffs[:, 0] == 0.0), (
                "Cannot multiply ln(x) terms in other"
            )
            mult_negative_coefs = np.zeros(
                (other.negative_coeffs.shape[0], other.negative_coeffs.shape[1])
            )

        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                for j in range(other.coeffs.shape[1]):
                    mult_coefs[seg, i + j] = (
                        mult_coefs[seg, i + j]
                        + self.coeffs[seg, i] * other.coeffs[seg, j]
                    )
                if other.negative_coeffs is not None:
                    for j in range(1, other.negative_coeffs.shape[1]):
                        index = i - j
                        if index >= 0:
                            # Still a positive index, includes 0 (cost terms)
                            mult_coefs[seg, index] += (
                                self.coeffs[seg, i] * other.negative_coeffs[seg, j]
                            )
                        else:
                            # negative index - put in -1*index of neg results
                            mult_negative_coefs[seg, -1 * index] += (
                                self.coeffs[seg, i] * other.negative_coeffs[seg, j]
                            )

            if self.negative_coeffs is not None:
                for i in range(1, self.negative_coeffs.shape[1]):
                    for j in range(other.coeffs.shape[1]):
                        index = j - i
                        if index >= 0:
                            mult_coefs[seg, index] += (
                                self.negative_coeffs[seg, i] * other.coeffs[seg, j]
                            )
                        else:
                            mult_negative_coefs[seg, -1 * index] += (
                                self.negative_coeffs[seg, i] * other.coeffs[seg, j]
                            )
                    if other.negative_coeffs is not None:
                        for j in range(1, other.negative_coeffs.shape[1]):
                            neg_index = i + j
                            mult_negative_coefs[seg, neg_index] += (
                                self.negative_coeffs[seg, i]
                                * other.negative_coeffs[seg, j]
                            )

        # TODO: handle non-overlapping breakpoints (first chop the
        # segments). Also implement do poly * const etc.

        mult_poly = PeicewisePolynomial(
            mult_coefs, mult_breakpoints, mult_negative_coefs
        )
        return mult_poly
