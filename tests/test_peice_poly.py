"""
Test cases for PeicewisePolynomial class

"""
import numpy as np
import numpy.testing as npt

import earth_model.peice_poly as pp

# FIXME: when run on it's own this does not
# test 32-33, 82, 111-129, 132-144 in peice_poly
# (although we get 100% coverage from earth_model
# tests). Also see comments below that indicate edge
# cases that need thinking about


def test_constant():
    """
    Check that a constant function gives allways gives it's value
    """
    poly = pp.PeicewisePolynomial(np.array([[2.0], [2.0]]),
                                  np.array([0.0, 0.5, 1.0]))
    assert poly(0.0) == 2.0
    assert poly(0.25) == 2.0
    assert poly(0.5) == 2.0
    assert poly(0.5, break_down=False) == 2.0
    assert poly(0.5, break_down=True) == 2.0
    assert poly(0.75) == 2.0
    assert poly(1.0) == 2.0


def test_step():
    """
    Check that two peicewise constants give the right values
    """
    poly = pp.PeicewisePolynomial(np.array([[2.0], [20.0]]),
                                  np.array([0.0, 0.5, 1.0]))
    assert poly(0.0) == 2.0
    assert poly(0.25) == 2.0
    assert poly(0.5) == 20.0
    assert poly(0.5, break_down=False) == 20.0
    assert poly(0.5, break_down=True) == 2.0
    assert poly(0.75) == 20.0
    assert poly(1.0) == 20.0
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
                        np.array([2.0, 2.0, 20.0, 20.0, 20.0]))
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                             break_down=False),
                        np.array([2.0, 2.0, 20.0, 20.0, 20.0]))
    npt.assert_allclose(poly(np.array([0.25, 0.5, 0.75, 1.0]),
                             break_down=True),
                        np.array([2.0, 2.0, 20.0, 20.0]))
    # What should we do on the edges
    # How should we report being outside the bounds
    # How should we handle the lower boundary with break_down=True


def test_deriv():
    poly = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                            [40.0, 30.0, 20.0]]),
                                  np.array([0.0, 2.0, 4.0]))
    expected_deriv_coefs = np.array([[3.0, 4.0], [30.0, 40.0]])
    calc_dpoly = poly.derivative()

    npt.assert_allclose(calc_dpoly.coeffs, expected_deriv_coefs)
    assert calc_dpoly(0.5) == 5.0
    assert calc_dpoly(3.0) == 150.0
    # What should we do on a breakpoint?


def test_antideriv():
    poly = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                            [40.0, 30.0, 20.0]]),
                                  np.array([0.0, 2.0, 4.0]))
    expected_antideriv_coefs = np.array([[0.0, 4.0, 3.0/2.0, 2.0/3.0],
                                         [0.0, 40.0, 30.0/2.0, 20.0/3.0]])
    calc_antideriv = poly.antiderivative()
    npt.assert_allclose(calc_antideriv.coeffs, expected_antideriv_coefs)


def test_integrate():
    poly = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                            [40.0, 30.0, 20.0]]),
                                  np.array([0.0, 2.0, 4.0]))
    antideriv = poly.antiderivative()

    expect_def_int_01 = 2/3 + 3/2 + 4
    calc_def_int_01 = antideriv.integrate(0, 1)
    npt.assert_allclose(calc_def_int_01, expect_def_int_01)

    expect_def_int_12 = ((2/3) * 8 + (3/2) * 4 + 4 * 2) - calc_def_int_01
    calc_def_int_12 = antideriv.integrate(1, 2)
    npt.assert_allclose(calc_def_int_12, expect_def_int_12)

    expect_def_int_24 = (((20/3) * 4**3 + (30/2) * 4**2 + 40 * 4) -
                         ((20/3) * 2**3 + (30/2) * 2**2 + 40 * 2))
    calc_def_int_24 = antideriv.integrate(2, 4)
    npt.assert_allclose(calc_def_int_24, expect_def_int_24)

    expect_def_int_14 = expect_def_int_12 + expect_def_int_24
    calc_def_int_14 = antideriv.integrate(1, 4)
    npt.assert_allclose(calc_def_int_14, expect_def_int_14)


def test_mult():
    poly1 = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                             [4.0, 3.0, 2.0]]),
                                   np.array([0.0, 2.0, 4.0]))
    poly2 = pp.PeicewisePolynomial(np.array([[1.0, 2.0, 4.0],
                                             [1.0, 2.0, 4.0]]),
                                   np.array([0.0, 2.0, 4.0]))
    # Why do we have the x^5 term (it's zero, but not needed...)
    expect_poly_mult = pp.PeicewisePolynomial(
        np.array([[4.0, 11.0, 24.0, 16.0, 8.0, 0.0],
                  [4.0, 11.0, 24.0, 16.0, 8.0, 0.0]]),
        np.array([0.0, 2.0, 4.0]))
    calc_poly_mult = poly1.mult(poly2)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)
