"""
Test cases for PREM model

Do we return the key parameters from the PREM
paper given the parameterisation? These are
more or less regression tests.
"""
import numpy as np
import numpy.testing as npt
import pytest

from prem4derg import PREM

# PREM data (from Table A.1. in intro to 
# physics of Earth's interior)
r = np.array([0.0, 300.0, 1200.0, 1400.0, 3100.0,
              3900.0, 5600.0, 6061.0, 6311.0]) # in km
p = np.array([3639.0, 3617.0, 3300.0, 3187.0, 1754.0,
              1118.0, 283.0, 102.0, 17.0]) * 0.1 # in GPa
g = np.array([0.0, 110.0, 432.0, 494.0, 981.0,
              1023.0, 1000.0, 994.0, 985.0]) / 100 # in m/s

def make_test_data(radius, data):
    """
    pytest's parametrize needs list of tuples.
    This function turns the numpy arrays above
    into such these.
    """
    test_data = []
    for tdi in zip(radius, data):
        test_data.append(tdi)
    return test_data


def test_earth_mass():
    """
    Check the mass of the Earth matches that
    given in the PREM paper
    """
    calculated_mass = PREM.mass(PREM.r_earth)
    expected_mass = 5.972e+24
    npt.assert_allclose(calculated_mass, expected_mass, atol=1.0E22)


@pytest.mark.parametrize("radius,expected_gravity", make_test_data(r,g) )
def test_prem_gravity(radius, expected_gravity):
    """
    A selection of cases from Table A.1 in Poirier's book
    'introduction to the physics of Earth's interior'
    """
    g_calc = PREM.gravity(radius)
    npt.assert_allclose(g_calc, expected_gravity, atol=0.05)


@pytest.mark.parametrize("radius,expected_pressure", make_test_data(r,p) )
def test_prem_pressure(radius, expected_pressure):
    """
    A selection of cases from Table A.1 in Poirier's book
    'introduction to the physics of Earth's interior'
    """
    p_calc = PREM.pressure(radius)
    # For the numerical integration we need a step size of ~1 km to get
    # reasonable performance, which leads to errors of ~1 GPa...
    npt.assert_allclose(p_calc, expected_pressure, atol=0.4)