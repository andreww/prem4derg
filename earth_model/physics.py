import numpy as np

from .const import G, R_EARTH
from .peice_poly import PeicewisePolynomial as PP


def calculate_vs(
    vs_poly: PP,
    r: float,
    t: float = 1,
    qm_poly: PP | None = None,
    break_down: bool = False,
) -> float | np.ndarray:
    """
    Evaluate s-wave velocity (in km/s) at radius r (in km).

    Optionally corrected for period (t), default is 1 s.
    """
    vs = vs_poly(r, break_down=break_down)
    if t != 1:
        if qm_poly is None:
            raise ValueError("qm_poly must be provided for attenuation correction")
        qm = qm_poly(r, break_down=break_down)
        vs = vs * (1.0 - ((np.log(t) / np.pi) * np.reciprocal(qm)))

    return vs


def calculate_vp(
    vp_poly: PP,
    r: float,
    t: float = 1,
    qk_poly: PP | None = None,
    qm_poly: PP | None = None,
    vs_poly: PP | None = None,
    break_down: bool = False,
) -> float | np.ndarray:
    """
    Evaluate p-wave velocity (in km/s) at radius r (in km).

    Optionally corrected for period (t), default is 1 s.
    """
    vp = vp_poly(r, break_down=break_down)
    if t != 1:
        if qk_poly is None or qm_poly is None or vs_poly is None:
            raise ValueError(
                "qk_poly, qm_poly and vs_poly must be provided for attenuation correction"
            )
        qm = qm_poly(r, break_down=break_down)
        qk = qk_poly(r, break_down=break_down)
        vs = vs_poly(r, break_down=break_down)
        e = (4 / 3) * ((vs / vp) ** 2)
        vp = vp * (
            1.0
            - (
                (np.log(t) / np.pi)
                * (((1.0 - e) * np.reciprocal(qk)) - e * np.reciprocal(qm))
            )
        )
    return vp


def calculate_bulk_modulus(
    vp_poly: PP, vs_poly: PP, density_poly: PP, r: float
) -> float:
    """
    Evaluate bulk modulus (in GPa) at radius r (in km)
    """
    vp = vp_poly(r) * 1000.0  # m/s
    mu = calculate_shear_modulus(vs_poly, density_poly, r)
    density = density_poly(r)
    return ((vp**2 * density) / 1e9) - mu


def calculate_shear_modulus(vs_poly: PP, density_poly: PP, r: float) -> float:
    """
    Evaluate shear modulus (in GPa) at radius r (in km)
    """
    vs = vs_poly(r) * 1000.0  # m/s
    density = density_poly(r)
    return (vs**2 * density) / 1.0e9


def calculate_mass(density_poly: PP, r: np.ndarray, r_inner=0.0):
    """
    Evaluate mass inside radius r (in km)
    """
    mass_poly = _setup_mass_poly(density_poly)
    if np.ndim(r) == 0:
        m = mass_poly.integrate(r_inner, r)
    else:
        m = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                m[i] = 0
            else:
                m[i] = mass_poly.integrate(r_inner, r[i])
    return m

def calculate_qkappa(qk_poly: PP, r: np.ndarray, break_down=False):
    return qk_poly(r, break_down=break_down)


def calculate_qshear(qm_poly: PP, r: np.ndarray, break_down=False):
    return qm_poly(r, break_down=break_down)

def calculate_density(density_poly: PP, r: np.ndarray, break_down=False):
    """
    Evaluate density in kg/m**3 at radii r (in km)
    """
    return density_poly(r, break_down=break_down)


def calculate_moi(density_poly: PP, r: np.ndarray, r_inner=0.0):
    """
    Evaluate moment of inertia inside radius r (in km)

    Return a tuple of moment of inertia (in kg m^2) and
    the moment of inertia factor (I/MR**2, dimensionless)
    which is 0.4 for a uniform density body, and decreases
    as the core becomes more dense than the crust/mantle.

    """
    moi_poly = _setup_moi_poly(density_poly)

    if np.ndim(r) == 0:
        moi = moi_poly.integrate(r_inner, r)
    else:
        moi = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                moi[i] = 0
            else:
                moi[i] = moi_poly.integrate(r_inner, r[i])

    r_in_m = r * 1000
    m = calculate_mass(density_poly, r, r_inner=r_inner)
    moif = moi / (m * (r_in_m**2))
    return moi, moif


def calculate_gravity(density_poly: PP, r: np.ndarray):
    gravity_poly = _setup_gravity_poly(density_poly)
    if np.ndim(r) == 0:
        if r == 0.0:
            return 0.0
        g = gravity_poly(r)
    else:
        g = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                g[i] = 0
            else:
                g[i] = gravity_poly(r[i])
    return g


def calculate_grav_potential(density_poly: PP, r: np.ndarray):
    """
    Evaluate the gravitational potential at radius r in J/kg
    """
    mass_poly = _setup_mass_poly(density_poly)
    if np.ndim(r) == 0:
        phi = -1 * mass_poly.integrate(0.0, r) / (r * 1000) * G
    else:
        phi = np.zeros_like(r)
        for i in range(r.size):
            if r[i] == 0:
                phi[i] = 0
            else:
                phi[i] = -1 * mass_poly.integrate(0.0, r[i]) / (r[i] * 1000) * G
    return phi


def calculate_pressure(density_poly: PP, r):
    """
    Evaluate pressure (in GPa) at radius r (in km)
    """
    pressure_poly = _setup_pressure_poly(density_poly)
    if np.ndim(r) == 0:
        p = pressure_poly.integrate(r, R_EARTH)
    else:
        p = np.zeros_like(r)
        for i in range(r.size):
            p[i] = pressure_poly.integrate(r[i], R_EARTH)
    return p


def _setup_mass_poly(density_poly: PP) -> PP:
    # setup polynomials for mass. This is 4*pi*\int rho(r)*r^2 dr
    bp = density_poly.breakpoints
    r2_params = np.tile(np.array([0.0, 0.0, 1000.0**3]), (bp.size - 1, 1))
    r2_poly = PP(r2_params, bp)
    mass_poly = density_poly.mult(r2_poly)
    mass_poly = mass_poly.antiderivative()
    #  integrating this gives mass:
    mass_poly.coeffs = mass_poly.coeffs * 4.0 * np.pi
    return mass_poly


def _setup_moi_poly(density_poly: PP) -> PP:
    # setup polynomials for MOI. This is 2/3*4*pi*\int rho(r)*r^4 dr
    bp = density_poly.breakpoints
    r4_params = np.tile(
        np.array([0.0, 0.0, 0.0, 0.0, 1000.0**5]), (bp.size - 1, 1)
    )
    r4_poly = PP(r4_params, bp)
    moi_poly = density_poly.mult(r4_poly)
    moi_poly = moi_poly.antiderivative()
    #   integrating this gives MOI:
    moi_poly.coeffs = moi_poly.coeffs * 4.0 * (2 / 3) * np.pi
    return moi_poly


def _setup_gravity_poly(density_poly: PP) -> PP:
    bp = density_poly.breakpoints
    r2_params = np.tile(np.array([0.0, 0.0, 1000.0**3]), (bp.size - 1, 1))
    r2_poly = PP(r2_params, bp)
    # Setup polynomial for gravity
    gravity_poly = density_poly.mult(r2_poly)
    # evaluate this to get int(rho.r^2 dr)
    gravity_poly = gravity_poly.integrating_poly()
    # constants outside integral
    gravity_poly.coeffs = gravity_poly.coeffs * 4.0 * np.pi * G
    over_r_sq_poly = PP(
        np.zeros((bp.size - 1, 1)),
        bp,
        np.zeros((bp.size - 1, 3)),
    )
    over_r_sq_poly.negative_coeffs[:, 2] = 1.0 / 1000.0**2
    gravity_poly = gravity_poly.mult(over_r_sq_poly)  # Mult by 1/r^2
    gravity_poly = gravity_poly  # Evaluate to get gravity at r
    return gravity_poly


def _setup_pressure_poly(density_poly: PP) -> PP:  # breakpoints not needed?
    # Setup polynomial for pressure:
    # integrate from r to r_earth to get pressure
    gravity_poly = _setup_gravity_poly(density_poly)
    pressure_poly = gravity_poly.mult(density_poly)
    pressure_poly = pressure_poly.antiderivative()
    # Pressure units (/1E9) and density units (*1000.0)
    pressure_poly.coeffs *= 1000.0 / 1.0e9
    pressure_poly.negative_coeffs *= 1000.0 / 1.0e9
    return pressure_poly
