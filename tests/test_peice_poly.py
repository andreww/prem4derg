"""
Test cases for PeicewisePolynomial class

"""
import numpy as np
import numpy.testing as npt

import earth_model.peice_poly as pp


def test_constant():
    """
    Check that a constant function gives allways gives its value
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
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
                        np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                             break_down=False),
                        np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
    npt.assert_allclose(poly(np.array([0.25, 0.5, 0.75, 1.0]),
                             break_down=True),
                        np.array([2.0, 2.0, 2.0, 2.0]))
    # FIXME - document and test behaviour at bounds


def test_quadratic():
    """
    Check that a quadratic function gives the correct value
    """
    poly = pp.PeicewisePolynomial(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                                  np.array([0.0, 0.5, 1.0]))
    assert poly(0.0) == 0.0
    assert poly(0.25) == 0.25**2
    assert poly(0.5) == 0.5**2
    assert poly(0.5, break_down=False) == 0.5**2
    assert poly(0.5, break_down=True) == 0.5**2
    assert poly(0.75) == 0.75**2
    assert poly(1.0) == 1.0
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
                        np.array([0.0, 0.25**2, 0.5**2, 0.75**2, 1.0]))
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                             break_down=False),
                        np.array([0.0, 0.25**2, 0.5**2, 0.75**2, 1.0]))
    npt.assert_allclose(poly(np.array([0.25, 0.5, 0.75, 1.0]),
                             break_down=True),
                        np.array([0.25**2, 0.5**2, 0.75**2, 1.0]))


def test_one_over_x():
    """
    Check that a 1/x function gives the correct value using a 
    quadratic for x less than 0.5
    """
    poly = pp.PeicewisePolynomial(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
                                  np.array([0.0, 0.5, 1.0]),
                                  c_neg=np.array([[0.0, 0.0], [0.0, 1.0]]))
    assert poly(0.0) == 0.0
    assert poly(0.25) == 0.25**2
    assert poly(0.5) == 1.0/0.5
    assert poly(0.5, break_down=False) == 1.0/0.5
    assert poly(0.5, break_down=True) == 0.5**2
    assert poly(0.75) == 1.0/0.75
    assert poly(1.0) == 1.0
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
                        np.array([0.0, 0.25**2, 1.0/0.5, 1.0/0.75, 1.0]))
    npt.assert_allclose(poly(np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                             break_down=False),
                        np.array([0.0, 0.25**2, 1.0/0.5, 1.0/0.75, 1.0]))
    npt.assert_allclose(poly(np.array([0.25, 0.5, 0.75, 1.0]),
                             break_down=True),
                        np.array([0.25**2, 0.5**2, 1.0/0.75, 1.0]))

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
    """
    Tests for polynomial derivatives.

    Note: y = 2x^2 + 3x + 4
          dy/dx = 4x + 3
          x = 0.5, dy/dx = 5
          x = 3, dy/dx = 15
          d2y/dx2 = 4
    And we can multiply everything by 10 and split the
    polynomial.
    """
    poly = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                            [40.0, 30.0, 20.0]]),
                                  np.array([0.0, 2.0, 4.0]))
    expected_deriv_coefs = np.array([[3.0, 4.0], [30.0, 40.0]])
    calc_dpoly = poly.derivative()

    npt.assert_allclose(calc_dpoly.coeffs, expected_deriv_coefs)
    assert calc_dpoly(0.5) == 5.0
    assert calc_dpoly(3.0) == 150.0

    expected_second_deriv_coefs = np.array([[4.0], [40.0]])
    calc_second_dpoly = calc_dpoly.derivative()
    npt.assert_allclose(calc_second_dpoly.coeffs, expected_second_deriv_coefs)
    assert calc_second_dpoly(0.5) == 4.0
    assert calc_second_dpoly(3.0) == 40.0
    # What should we do on a breakpoint?


def test_recip_deriv():
    """
    Tests for polynomial derivatives including terms like 1/x.

    Note: y = 4x + 3 + 2/x + 3/x^2
          dy/dx = 4 - 2/x^2 - 6/x^3
          x = 0.5, dy/dx = -52
          x = 3, dy/dx = 3.55555...
          d2y/dx2 = 4
    And we can multiply everything by 10 and split the
    polynomial.
    """
    poly = pp.PeicewisePolynomial(np.array([[3.0, 4.0],
                                            [30.0, 40.0]]),
                                  np.array([0.0, 2.0, 4.0]),
                                  c_neg=np.array([[0.0, 2.0, 3.0],
                                                  [0.0, 20.0, 30.0]]))
    expected_deriv_coefs = np.array([[4.0], [40.0]])
    expected_neg_deriv_coeffs = np.array([[0.0, 0.0, -2.0, -6.0],
                                          [0.0, 0.0, -20.0, -60.0]])
    calc_dpoly = poly.derivative()
    npt.assert_allclose(calc_dpoly.coeffs, expected_deriv_coefs)
    npt.assert_allclose(calc_dpoly.negative_coeffs, expected_neg_deriv_coeffs)

    assert calc_dpoly(0.5) == -52.0
    npt.assert_allclose(calc_dpoly(3.0), 3.55555555555*10.0)


def test_antideriv():
    """
    Tests for polynomial antiderivatives.

    Note: dy/dx = 2x^2 + 3x + 4
          y = 2/3x^3 +3/2x^2 + 4x + C
    And we can multiply everything by 10 and split the
    polynomial.
    """
    poly = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                            [40.0, 30.0, 20.0]]),
                                  np.array([0.0, 2.0, 4.0]))
    expected_antideriv_coefs = np.array([[0.0, 4.0, 3.0/2.0, 2.0/3.0],
                                         [0.0, 40.0, 30.0/2.0, 20.0/3.0]])
    calc_antideriv = poly.antiderivative()
    npt.assert_allclose(calc_antideriv.coeffs, expected_antideriv_coefs)
    # should this really be exposed in the api?


def test_recip_antideriv():
    """
    Tests for polynomial antiderivatives with 1/x type terms.

    Note: dy/dx = 4 - 2/x^2 - 6/x^3
          y = 4x + 2/x + 3/x^2 + C
    And we can multiply everything by 10 and split the
    polynomial.
    """
    poly = pp.PeicewisePolynomial(np.array([[4.0],
                                            [40.0]]),
                                  np.array([0.0, 2.0, 4.0]),
                                  c_neg=np.array([[0.0, 0.0, -2.0, -6.0],
                                                  [0.0, 0.0, -20.0, -60.0]]))
    expected_int_coefs = np.array([[0.0, 4.0], [0.0, 40.0]])
    expected_neg_int_coeffs = np.array([[0.0, 2.0, 3.0],
                                        [0.0, 20.0, 30.0]])
    
    calc_dpoly = poly.antiderivative()
    npt.assert_allclose(calc_dpoly.coeffs, expected_int_coefs)
    npt.assert_allclose(calc_dpoly.negative_coeffs, expected_neg_int_coeffs)


def test_log_deriv_int():
    """
    Tests for polynomial derivatives including terms like 1/x.

    Note: y = 4x + 3 + 2/x + 3/x^2 + 5 ln(|x|)
          dy/dx = 4 - 2/x^2 - 6/x^3 + 5/x
          x = 0.5, dy/dx = -42
          x = 3, dy/dx = 3.55555... + 5/3
    And we can multiply everything by 10 and split the
    polynomial.
    """
    poly = pp.PeicewisePolynomial(np.array([[3.0, 4.0],
                                            [30.0, 40.0]]),
                                  np.array([0.0, 2.0, 4.0]),
                                  c_neg=np.array([[5.0, 2.0, 3.0],
                                                  [50.0, 20.0, 30.0]]))
    expected_deriv_coefs = np.array([[4.0], [40.0]])
    expected_neg_deriv_coeffs = np.array([[0.0, 5.0, -2.0, -6.0],
                                          [0.0, 50.0, -20.0, -60.0]])
    calc_dpoly = poly.derivative()
    npt.assert_allclose(calc_dpoly.coeffs, expected_deriv_coefs)
    npt.assert_allclose(calc_dpoly.negative_coeffs, expected_neg_deriv_coeffs)

    assert calc_dpoly(0.5) == -42.0
    npt.assert_allclose(calc_dpoly(3.0), (3.55555555555+(5/3))*10.0)

    # Do we get back to what we started with (without the constant)?
    calc_ipoly = calc_dpoly.antiderivative()
    npt.assert_allclose(calc_ipoly.coeffs, np.array([[0.0, 4.0], [0.0, 40.0]]))
    npt.assert_allclose(calc_ipoly.negative_coeffs, np.array([[5.0, 2.0, 3.0], [50.0, 20.0, 30.0]]))

    # Check the values
    npt.assert_allclose(calc_ipoly(0.5), 2.0 + (5.0 * np.log(0.5)) + 4.0 + 12.0)
    npt.assert_allclose(calc_ipoly(3.0), (40.0*3.0) + (50.0 * np.log(3.0)) + (20.0/3.0) + (30.0/9.0))

def test_integrate():
    """
    Tests for polynomial integration.

    Note: y = 2x^2 + 3x + 4
          int(y)_0^1 = 2/3 + 3/4 + 4
          int(y)_1^2 = (8*2/3 + 4*3/4 + 8) - (2/3 + 3/4 + 4)
          etc...
    And we can multiply everything by 10 and split the
    polynomial with addition over breakpoints.
    """
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
    # what if we reverse order of bounds?


def test_mult():
    """
    Tests for polynomial multiplication.

    Note: (2x^2 + 3x + 4) * (4x^2 + 2x + 1)
        = 8x^4 + 16x^3 + 24x^2 + 11x +4
    and we split this polynomial in two to excercise the
    breakpoint checking.
    """
    poly1 = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                             [4.0, 3.0, 2.0]]),
                                   np.array([0.0, 2.0, 4.0]))
    poly2 = pp.PeicewisePolynomial(np.array([[1.0, 2.0, 4.0],
                                             [1.0, 2.0, 4.0]]),
                                   np.array([0.0, 2.0, 4.0]))
    expect_poly_mult = pp.PeicewisePolynomial(
        np.array([[4.0, 11.0, 24.0, 16.0, 8.0],
                  [4.0, 11.0, 24.0, 16.0, 8.0]]),
        np.array([0.0, 2.0, 4.0]))
    calc_poly_mult = poly1.mult(poly2)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)

def test_mult2():
    """
    Tests for polynomial multiplication.

    Includes 1/x terms etc. 
    Note: (2x^2 + 3x + 4  + 3/x + 4/x^2) * (4x^2 + 2x + 1 + 2/x^2)
        = 8x^4 + 16x^3 + 24x^2 + 23x + 14 + 9/x + 8/x^2 + 6/x^3 + 8/x^4
    and we split the polynomials in two to excercise the
    breakpoint checking. Also check that order does not matter and
    we can square both polynomials
    """
    poly1 = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                             [4.0, 3.0, 2.0]]),
                                   np.array([0.0, 2.0, 4.0]),
                                   np.array([[0.0, 3.0, 4.0],
                                             [0.0, 3.0, 4.0]]))
    poly2 = pp.PeicewisePolynomial(np.array([[1.0, 2.0, 4.0],
                                             [1.0, 2.0, 4.0]]),
                                   np.array([0.0, 2.0, 4.0]),
                                   np.array([[0.0, 0.0, 2.0],
                                             [0.0, 0.0, 2.0]]))
    expect_poly_mult = pp.PeicewisePolynomial(
        np.array([[30.0, 23.0, 24.0, 16.0, 8.0],
                  [30.0, 23.0, 24.0, 16.0, 8.0]]),
        np.array([0.0, 2.0, 4.0]),
        np.array([[0.0, 17.0, 12.0, 6.0, 8.0],
                  [0.0, 17.0, 12.0, 6.0, 8.0]]))
    calc_poly_mult = poly1.mult(poly2)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)
    npt.assert_allclose(calc_poly_mult.negative_coeffs, expect_poly_mult.negative_coeffs)

    # backwards
    calc_poly_mult = poly2.mult(poly1)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)
    npt.assert_allclose(calc_poly_mult.negative_coeffs, expect_poly_mult.negative_coeffs)

    # Square poly 1
    expect_poly_mult = pp.PeicewisePolynomial(
        np.array([[50.0, 36.0, 25.0, 12.0, 4.0],
                  [50.0, 36.0, 25.0, 12.0, 4.0]]),
        np.array([0.0, 2.0, 4.0]),
        np.array([[0.0, 48.0, 41.0, 24.0, 16.0],
                  [0.0, 48.0, 41.0, 24.0, 16.0]]))
    calc_poly_mult = poly1.mult(poly1)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)
    npt.assert_allclose(calc_poly_mult.negative_coeffs, expect_poly_mult.negative_coeffs)

def test_mult3():
    """
    Tests for polynomial multiplication.

    Where only one includes 1/x terms etc. 
    Note: (2x^2 + 3x + 4  + 3/x + 4/x^2) * (4x^2 + 2x + 1)
        = 8x^4 + 16x^3 + 24x^2 + 23x + 26 + 11/x + 4/x^2
    and we split the polynomials in two to excercise the
    breakpoint checking. Also check that order does not matter
    """
    poly1 = pp.PeicewisePolynomial(np.array([[4.0, 3.0, 2.0],
                                             [4.0, 3.0, 2.0]]),
                                   np.array([0.0, 2.0, 4.0]),
                                   np.array([[0.0, 3.0, 4.0],
                                             [0.0, 3.0, 4.0]]))
    poly2 = pp.PeicewisePolynomial(np.array([[1.0, 2.0, 4.0],
                                             [1.0, 2.0, 4.0]]),
                                   np.array([0.0, 2.0, 4.0]))
    expect_poly_mult = pp.PeicewisePolynomial(
        np.array([[26.0, 23.0, 24.0, 16.0, 8.0],
                  [26.0, 23.0, 24.0, 16.0, 8.0]]),
        np.array([0.0, 2.0, 4.0]),
        np.array([[0.0, 11.0, 4.0],
                  [0.0, 11.0, 4.0]]))

    calc_poly_mult = poly1.mult(poly2)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)
    npt.assert_allclose(calc_poly_mult.negative_coeffs, expect_poly_mult.negative_coeffs)

    # backwards
    calc_poly_mult = poly2.mult(poly1)
    npt.assert_allclose(calc_poly_mult.coeffs, expect_poly_mult.coeffs)
    npt.assert_allclose(calc_poly_mult.negative_coeffs, expect_poly_mult.negative_coeffs)