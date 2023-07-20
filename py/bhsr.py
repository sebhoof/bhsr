#####################################
#  BHSR rates and related functions #
#####################################

import warnings
import numpy as np

from math import factorial, prod
from fractions import Fraction
from scipy.optimize import root_scalar
from superrad.ultralight_boson import UltralightBoson
from .constants import inv_tSR, mP_in_GeV, inv_eVs, yr_in_s
from .kerr_bh import *

## BHSR rates using Dettweiler's approximation

def omega(mu: float, mbh: float, n: int, l: int, m: int) -> float:
    """
    Calculates the frequency of the superradiant mode for a given set of quantum numbers.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number (currently not used).
        m (int): Magnetic quantum number (currently not used).

    Returns:
        float: The frequency of the superradiant mode in eV.

    Notes:
        - This function uses the LO approximation for the energy of the state |n,l,m>.
    """
    x = alpha(mu, mbh)/n
    return mu*(1.0 - 0.5*x*x)

def c_nl(n: int, l: int) -> Fraction:
    """
    Helper function for computing a factor in GammaSR_nlm.

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.

    Returns:
        Fraction: Multiplication factor.
    """
    x = Fraction(factorial(n+l) * 2**(4*l+1), factorial(n-l-1) * n**(2*l+4) )
    y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
    return x*y*y

def c_nl_float(n: int, l: int) -> float:
    """
    Helper function for computing a factor in GammaSR_nlm (returns float).

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.

    Returns:
        float: Multiplication factor.
    """
    c_nl_fr = c_nl(n, l)
    return 1.0*c_nl_fr

def GammaSR_nlm(mu: float, mbh: float, astar: float, n: int, l: int, m: int) -> float:
    """
    Calculate the superradiance rate for a bosonic cloud around a Kerr black hole.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter.
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        m (int): Magnetic quantum number.

    Returns:
        float: Superradiance rate in eV

    Notes:
        - This formula uses (the corrected?) Dettweiler approximation.
        - We use $n \equiv \bar{n} = n - l - 1$.
        - Refs.: Eqs. (5), (13-15) in https://arxiv.org/pdf/2009.07206.pdf; also https://arxiv.org/pdf/1501.06570.pdf, https://arxiv.org/pdf/1411.2263.pdf
        - Differences to Eq. (14) in https://arxiv.org/pdf/1805.02016.pdf
    """
    al = alpha(mu, mbh)
    marp = al*(1 + np.sqrt(1-astar*astar))
    x = m*astar - 2*marp
    y = al**(4*(l+1))
    factors = [(k*k)*(1.0-astar*astar) + x**2 for k in range(1,l+1)]
    return c_nl_float(n, l) * mu*x * prod(factors) * al**(4*(l+1))

## BHSR rates using the "non-relativistic approximation"

def c_nl_nr(n: int, l: int) -> Fraction:
    """
    Helper function for computing a factor in GammaSR_nlm_nr.

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.

    Returns:
        Fraction: Multiplication factor.
    """
    x = Fraction(factorial(2*l+n+1) * 2**(4*l+2), factorial(n) * (l+n+1)**(2*l+4) )
    y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
    return x*y*y

def c_nl_nr_float(n: int, l: int) -> float:
    """
    Helper function for computing a factor in GammaSR_nlm_nr (returns float).

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.

    Returns:
        float: Multiplication factor.
    """
    c_nl_fr = c_nl_nr(n, l)
    return 1.0*c_nl_fr

def GammaSR_nlm_nr(mu: float, mbh: float, astar: float, n: int, l: int, m: int) -> float:
    """
    Calculate the superradiance rate for a bosonic cloud around a Kerr black hole,
    using the "non-realtivisic approximation."

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter.
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        m (int): Magnetic quantum number.

    Returns:
        float: Superradiance rate in eV

    Notes:
        - Ref.: see Fig. 5 in https://arxiv.org/pdf/1004.3558.pdf
    """
    al = alpha(mu, mbh)
    marp = al*(1 + np.sqrt(1-astar*astar))
    x = 2.0*(0.5*m*astar - marp)
    factors = [(k*k)*(1.0-astar*astar) + x*x for k in range(1,l+1)]
    return  mu*x * al**(4*(l+1)) * c_nl_nr_float(n, l) * prod(factors)

## BHSR rates from superrad Python package

bc0 = UltralightBoson(spin=0, model="relativistic")
def GammaSR_nlm_superrad(mu: float, mbh: float, astar: float, bc: UltralightBoson) -> float:
    """
    Calculate the superradiance rate for a bosonic cloud around a Kerr black hole,
    using the superrad Python package (https://bitbucket.org/weast/superrad)

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter.
        bc (UltralightBoson, optional): Type of boson to consider (default: relativistic scalar)

    Returns:
        float: Superradiance rate in eV

    Notes:
        - Ref.: https://arxiv.org/pdf/2211.03845.pdf
    """
    try:
        wf = bc.make_waveform(mbh, astar, mu, units="physical")
        return 2.0*np.pi*inv_eVs/wf.cloud_growth_time()
    except ValueError:
        return 0

# Compute spindown rate according to quasi-equilibrium approximation
# Follow O. Simon's unpublished notes
def GammaSR_322xBH_211x211(ma, mbh, astar, fa):
    al = alpha(ma, mbh)
    return 4.3e-7 * ma*(1.0 + np.sqrt(1.0 - astar*astar))*pow(al, 11)*pow(mP_in_GeV/fa, 4)

def GammaSR_211xinf_322x322(ma, mbh, fa):
    al = alpha(ma, mbh)
    return 1.1e-8 * ma*pow(al, 8)*pow(mP_in_GeV/fa, 4)

def n_eq_211(ma, mbh, astar, fa):
    sr0 = GammaSR_nlm(ma, mbh, astar, 2, 1, 1)
    sr_3b22 = GammaSR_322xBH_211x211(ma, mbh, astar, fa)
    sr_2i33 = GammaSR_211xinf_322x322(ma, mbh, fa)
    return 2.0*np.sqrt(sr0*sr_2i33/3.0)/sr_3b22

def GammaSR_nlm_eq(ma, mbh, astar, fa):
    sr0 = GammaSR_nlm(ma, mbh, astar, 2, 1, 1)
    neq = n_eq_211(ma, mbh, astar, fa)
    return neq*sr0

def n_max(mbh: float, da: float = 0.1) -> float:
    """
    Calculate the maximum value of the boson occupation number of the superradiant cloud.

    Parameters:
        mbh (float): Black hole mass in Msol.
        da (float, optional): Difference of dimensionless initial and final spin (default: 0.1).

    Returns:
        float: The maximum value of the boson occupation number.

    Notes:
        - Ref. Eq. (8) in https://arxiv.org/pdf/1411.2263.pdf
    """
    return 1e76 * (da/0.1) * (mbh/10)**2

def is_sr_mode(mu, mbh, astar, tbh, n, l, m):
    nm = n_max(mbh)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = GammaSR_nlm(mu, mbh, astar, n, l, m) > inv_t*np.log(nm)
    if np.isnan(res):
        res = 0
    return res

def is_sr_mode_min(mu, min_sr_rate, mbh, tbh):
    nm = n_max(mbh)
    inv_t = inv_eVs / (yr_in_s*tbh)
    return min_sr_rate > inv_t*np.log(nm)
   
## Regge slope

def compute_regge_slopes(mu: float, mbh_vals: list[float], states: list[tuple[int, int, int]], inv_tSR: float = inv_tSR) -> np.ndarray:
    """
    Compute the Regge slopes/curves for specific quantum states.

    Parameters:
        mu (float): Boson mass in eV.
        mbh_vals (list[float]): List of black hole masses in Msol.
        states (list[tuple[int, int, int]]): List of quantum states (n, l, m).
        inv_tSR (float, optional): Inverse of the superradiance timescale (default: inv_tSR).

    Returns:
        np.ndarray: Array of dimensionless black hole spins corresponding to mbh_vals.

    Notes:
        - The Regge slope is defined as the minimum value of the dimensionless spin parameter 'a' at which the superradiance rate equals the inverse of the superradiance timescale.
        - Note that the root finding may fail if no Regge slope exists. In this case, we set the corresponding BH spin value = NAN.
    """
    a_min_vals = []
    foo = lambda a, mbh, n, l, m: GammaSR_nlm(mu, mbh, a, n, l, m) - inv_tSR
    for mbh in mbh_vals:
        temp = []
        for s in states:
            with warnings.catch_warnings(record=True) as w:
                res = root_scalar(foo, x0=0, x1=0.5, args=(mbh, *s))
                a_root = res.root if len(w) == 0 else np.nan
                temp.append(a_root)
        a_min_vals.append(temp)
    return np.array(a_min_vals)


def compute_regge_slopes_given_rate(mu: float, mbh_vals: list[float], sr_function: callable, inv_tSR: float = inv_tSR) -> np.ndarray:
    """
    Compute the Regge slopes given a superradiance rate function.

    Parameters:
        mu (float): Boson mass in eV.
        mbh_vals (list[float]): List of black hole masses in Msol.
        sr_function (callable): Superradiance rate function with signature sr_function(mu, mbh, astar)
        inv_tSR (float, optional): Inverse of the superradiance timescale (default: inv_tSR).

    Returns:
        np.ndarray: Array of Regge slopes.

    Notes:
        - The Regge slope is defined as the minimum value of the dimensionless spin parameter 'a' at which the superradiance rate equals the inverse of the superradiance timescale.
        - Note that the root finding may fail if no Regge slope exists, or the user-defined function may produce an error, e.g. because of large spins or difficult BH masses.
          In any such case, we set the corresponding BH spin value = NAN.
    """
    a_min_vals = []
    foo = lambda a, mbh: sr_function(mu, mbh, a) - inv_tSR
    for mbh in mbh_vals:
        with warnings.catch_warnings(record=True) as w:
            try:
                res = root_scalar(foo, bracket=[0.01, 0.99], args=(mbh))
                a_root = res.root if len(w) == 0 else np.nan
            except ValueError:
                a_root = np.nan
        a_min_vals.append(a_root)
    return np.array(a_min_vals)
