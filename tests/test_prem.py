"""
Test cases for PREM model

Do we return the key parameters from the PREM
paper given the parameterisation? These are
more or less regression tests.
"""
import numpy.testing as npt

from earth_model.earth_model import Prem


def test_earth_mass():
    prem = Prem()
    calculated_mass = prem.mass(prem.r_earth)
    expected_mass = 5.972e+24
    npt.assert_allclose(calculated_mass, expected_mass, atol=1.0E22)
