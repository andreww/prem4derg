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
