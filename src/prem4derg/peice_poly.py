#!/usr/bin/env python
# coding=utf8
"""
Peicewise polynomials like PREM

"""

import numpy as np


class PeicewisePolynomial:
    """
    Piecewise Polynomials a different way

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

    def __init__(self, coeffs: np.ndarray, breakpoints: np.ndarray, neg_coeffs: np.ndarray | None = None) -> None:
        if breakpoints.ndim != 1:
            raise ValueError("Breakpoints must be 1D")
        self.breakpoints = breakpoints

        self.coeffs = _standardize_coeffs_shapes(coeffs)
        
        if neg_coeffs is not None:
            self.negative_coeffs = _standardize_coeffs_shapes(neg_coeffs)
        else:
            self.negative_coeffs = None

    def __call__(self, xp: float | np.ndarray, break_down: bool = False) -> float | np.ndarray:
        """
        Vectorised evaluation.

        break_down=False: intervals [b_i, b_{i+1})
        break_down=True:  intervals (b_i, b_{i+1}]
        Special case: x == last breakpoint -> last segment.
        """
        xarr = np.asarray(xp, dtype=float)
        scalar_input = xarr.ndim == 0
        xflat = xarr.ravel()

        # Segment indices (-1 means out of domain)
        seg_idx = self._segment_indices(xflat, break_down)

        if np.any(seg_idx < 0):
            raise ValueError("Some evaluation points lie outside breakpoint domain")

        # Prepare output
        out = np.zeros_like(xflat, dtype=float)

        n_segments = self.breakpoints.size - 1
        pos_degree = self.coeffs.shape[1]
        powers = np.arange(pos_degree)

        for seg in range(n_segments):
            mask = seg_idx == seg
            if not np.any(mask):
                continue
            xs = xflat[mask]
            # Positive-power part
            pos_coef = self.coeffs[seg]  # shape (pos_degree,)
            # xs[:, None] ** powers -> shape (n_pts_seg, pos_degree)
            out[mask] += (pos_coef * (xs[:, None] ** powers)).sum(axis=1)

            # Negative / ln terms
            if self.negative_coeffs is not None:
                neg_coef = self.negative_coeffs[seg]
                if np.any(neg_coef != 0.0):
                    # ln term (index 0)
                    if neg_coef[0] != 0.0:
                        if np.any(xs == 0.0):
                            raise ValueError("ln(|x|) undefined at x = 0")
                        out[mask] += neg_coef[0] * np.log(np.abs(xs))
                    # reciprocal terms (indices >=1)
                    if neg_coef.size > 1:
                        rec_indices = np.arange(1, neg_coef.size)
                        nz = neg_coef[1:] != 0.0
                        if np.any(nz):
                            if np.any((xs == 0.0)):
                                # Division by zero for any non-zero reciprocal coefficient
                                if np.any(neg_coef[1:] != 0.0):
                                    raise ZeroDivisionError("Division by zero in 1/x^i term")
                            # Compute only for non-zero coefficients
                            active_i = rec_indices[nz]
                            active_c = neg_coef[1:][nz]
                            # Sum_{i} c_i / x^i
                            # xs[:, None] ** (-active_i) -> shape (n_pts, n_active_powers)
                            # active_c -> shape (n_active_powers,)
                            out[mask] += (active_c * (xs[:, None] ** (-active_i))).sum(axis=1)

        if scalar_input:
            return float(out[0])
        return out.reshape(xarr.shape)

    def _segment_indices(self, x: np.ndarray, break_down: bool) -> np.ndarray:
        """
        Return segment indices for each x.
        -1 indicates out-of-domain.
        """
        b = self.breakpoints
        n_segments = b.size - 1
        seg_idx = np.empty_like(x, dtype=int)

        if break_down:
            # (b_i, b_{i+1}] -> use left-open search
            idx = np.searchsorted(b, x, side="left") - 1
        else:
            # [b_i, b_{i+1}) -> use right-side search
            idx = np.searchsorted(b, x, side="right") - 1

        # Special case: x == last breakpoint -> last segment
        last_bp = b[-1]
        at_last = x == last_bp
        if np.any(at_last):
            idx[at_last] = n_segments - 1  # last segment

        # Out of domain: x < b[0] or x > b[-1] always invalid
        invalid = (x < b[0]) | (x > b[-1])
        # For break_down True also exclude x == b[0]
        if break_down:
            invalid |= (x == b[0])

        # Clamp valid indices; mark invalid as -1
        seg_idx[:] = idx
        seg_idx[invalid] = -1
        return seg_idx

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

    def mult(self, other: "PeicewisePolynomial") -> "PeicewisePolynomial":
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


def _standardize_coeffs_shapes(coeffs: np.ndarray) -> np.ndarray:
    if coeffs.ndim == 1:
        coeffs = np.expand_dims(coeffs, axis=1)
        coeffs = np.append(coeffs, np.zeros_like(coeffs), axis=1)
    elif coeffs.ndim == 2:
        pass
    else:
        raise ValueError("Coefficients must be 1D or 2D")
    return coeffs