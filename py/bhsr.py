#####################################
#  BHSR rates and related functions #
#####################################

import warnings
import numpy as np

from .constants import *
from .kerr_bh import *
from fractions import Fraction
from math import factorial, prod
from scipy.optimize import root_scalar
from scipy.special import gamma
from superrad.ultralight_boson import UltralightBoson


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
    Helper function for computing a factor in GammaSR_nlm_nr.

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.

    Returns:
        Fraction: Multiplication factor.

    Notes:
        - The alternative quantum number n', with n = n' + l + 1, is used in some publications.
    """
    x = Fraction(factorial(n+l) * pow(2, 4*l+2), factorial(n-l-1) * pow(n, 2*l+4))
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
    return 1.0*c_nl(n, l)

## BHSR rates using the "non-relativistic approximation"

def GammaSR_nlm_nr(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> float:
    """
    Calculate the superradiance rate for a bosonic cloud around a Kerr black hole,
    using the "non-realtivisic approximation."

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter (astar = a/rg).
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        m (int): Magnetic quantum number.

    Returns:
        float: Superradiance rate in eV

    Notes:
        - See Eq. (18) in https://arxiv.org/pdf/1004.3558.pdf
        - Fig. 5 in https://arxiv.org/pdf/1004.3558.pdf
        - The authors of https://arxiv.org/pdf/1004.3558.pdf also use a semi-analytical method (see also https://arxiv.org/pdf/0912.1780.pdf)
    """
    al = alpha(mu, mbh)
    x = 1.0 - astar*astar
    murp = al*(1 + np.sqrt(x)) # = mu*rg*rp
    y = m*astar - 2*murp
    factors = [(k*k)*x + y*y for k in range(1,l+1)]
    return  mu*y*c_nl_float(n, l)*prod(factors)*pow(al, 4*l+4)

## BHSR rates using corrected Dettweiler's formula + relativisitc correction (based on https://arxiv.org/pdf/2201.10941.pdf)

def omega0_bxzh(mu: float, mbh: float, n: int) -> float:
    n2 = n*n
    al = alpha(mu, mbh)
    al2 = al*al
    x = 2*al2/(n2 + 4*al2 + n*np.sqrt(n2 + 8*al2))
    return mu*np.sqrt(1.0 - x)

def omega1_bxzh(mu: float, mbh: float, n: int) -> float:
    om0 = omega0_bxzh(mu, mbh, n)
    om02 = om0*om0
    mu2 = mu*mu
    al = alpha(mu, mbh)
    al2 = al*al
    x = 1.0 + 4*al2*(2*om02/mu2 - 1.0)/(n*n)
    return (mu2 - om02)/(n*om0*x)

def c_nl_bxzh(n: int, l: int) -> Fraction:
    x = Fraction(factorial(n+l) * pow(2, 4*l+2), factorial(n-l-1))
    y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
    return x*y*y

def c_nl_bxzh_float(n: int, l: int) -> float:
    return 1.0*c_nl_bxzh(n, l)

def GammaSR_nlm_bxzh(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> float:
    """
    Calculate the superradiance rate for a bosonic cloud around a Kerr black hole,
    using the corrected Dettweiler approximation.

    Returns:
        float: Superradiance rate in eV

    Notes:
        - Ref.: https://arxiv.org/pdf/2201.10941.pdf
    """
    om0 = omega0_bxzh(mu, mbh, n)
    al = alpha(mu, mbh)
    z = pow(al*al*(1.0-om0*om0/(mu*mu)), l+0.5)
    om1 = omega1_bxzh(mu, mbh, n)
    c_nl = c_nl_bxzh_float(n, l)
    x = 1.0-astar*astar
    rp = r_plus(mbh, astar)
    y = m*astar - 2*rp*om0
    factors = [k*k*x + y*y for k in range(1,l+1)]
    return y*z*c_nl*prod(factors)*om1

def gam_pq_bxzh(p: float, q: float, eps: float, n: int, l: int) -> float:
    lp = l + eps
    ip = p*1j
    twolp = 2*lp
    g1 = gamma(twolp + 1)
    g2 = gamma(twolp + 2)
    g2n = gamma(twolp + 1 + n - l)
    gpmeps = gamma(1 + 2*eps)*gamma(1 - 2*eps)
    x1 = lp + 1 + ip
    x2 = np.sqrt(q - p*p + 0j)
    gabs = np.abs(gamma(x1 + x2)*gamma(x1 - x2))
    gmix = gamma(1.0 - ip - eps + x2)*gamma(1.0 - ip - eps - x2)
    gmix *= gamma(1.0 + ip + eps + x2)*gamma(1.0 + ip + eps - x2)
    num = g2n*gpmeps*gabs*gabs*pow(2, 4*lp + 2)
    denom = factorial(n-l-1)*g1*g1*g2*g2*gmix
    return num/denom

def omega_nlm_bxzh(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> float:
    om0 = omega0_bxzh(mu, mbh, n)
    om1 = omega1_bxzh(mu, mbh, n)
    al = alpha(mu, mbh)
    al2 = al*al
    eps = -8.0*al2/(2*l+1)
    lp = l + eps
    if lp < 0:
        return 0, 0
    rp = r_plus(mbh, astar)
    rG = rg(mbh)
    x = np.sqrt(1.0 - astar*astar)
    y = mu*mu - om0*om0
    p = -0.5*(m*astar - 2.0*rp*om0)/x
    q = 4*om0*p*rG - 2*(3.0 - x)*al2
    gam_terms = gam_pq_bxzh(p, q, eps, n, l)
    kappab_term = pow(rG*rG*x*x*y, lp+0.5)
    delta1 = 0.5*(q/eps - eps - p*2j)*kappab_term*gam_terms
    om = om0 + (eps + delta1)*om1
    return np.real(om), np.imag(om)

## BHSR rates from superrad Python package

def GammaSR_nlm_superrad(mu: float, mbh: float, astar: float, bc: UltralightBoson, m: int) -> float:
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
        """
        evo_type can take values evo_type="full" or evo_type="matched".
        The "matched" evolution assumes that the boson cloud decays solely through gravitational radiation after reach its peak mass (by definition, t=0), and matches this on to a exponentially growing solution with constant growth rate before the peak mass is obtained (t<0).
        Hence, it is not very accurate around t=0, and in particular the derivative of the frequency will be discontinuous at this point.
        The "full" evolution option numerically integrates the ordinary differential equations describing the cloud mass and black hole mass and spin, and hence is more accurate for scenarios when the signal before the time when the cloud reaches saturation makes a non-negligible contribution.
        However, it is significantly more computationally expensive, and the numerical integration is not always robust. This option should currently be considered experimental.
        Details and a comparison of these methods can be found in the main paper.
        """;
        # wf = bc.make_waveform(mbh, astar, mu, units="physical", evo_type="matched")
        # return inv_eVs/wf.efold_time()# .cloud_growth_time()
        al = alpha(mu, mbh)
        rG = rg(mbh)
        return bc._cloud_model.omega_imag(m, al, astar)/rG
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
    sr0 = GammaSR_nlm_nr(ma, mbh, astar, 2, 1, 1)
    sr_3b22 = GammaSR_322xBH_211x211(ma, mbh, astar, fa)
    sr_2i33 = GammaSR_211xinf_322x322(ma, mbh, fa)
    return 2.0*np.sqrt(sr0*sr_2i33/3.0)/sr_3b22

def GammaSR_nlm_eq(ma, mbh, astar, fa):
    sr0 = GammaSR_nlm_nr(ma, mbh, astar, 2, 1, 1)
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
    res = GammaSR_nlm_nr(mu, mbh, astar, n, l, m) > inv_t*np.log(nm)
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
    foo = lambda a, mbh, n, l, m: GammaSR_nlm_nr(mu, mbh, a, n, l, m) - inv_tSR
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

def is_box_allowed_211(mu: float, invf: float, bh_data: list[int, ...], sr_function: callable, sigma_level: float = 2):
    _, tbh, mbh, mbh_err, a, _, a_err_m = bh_data
    # Coservative approach by choosing the shortest BH time scale
    tbh = min(tEddington_in_yr, tbh)
    mbh_p, mbh_m = mbh+sigma_level*mbh_err, max(0,mbh-sigma_level*mbh_err)
    a_m = max(0, a-sigma_level*a_err_m)
    for mm in np.linspace(mbh_m, mbh_p, 25):
        inv_t = inv_eVs / (yr_in_s*tbh)
        # Check SR condition (l = 1)
        if (alpha(mu, mm) <= 0.5):
            sr0 = sr_function(mu, mbh, a_m)
            if sr0 > inv_t:
                sr = sr0*n_eq_211(mu, mm, a_m, 1/invf)
                if sr > inv_t:
                    return 0
                else:
                    return 1
            else:
                return 1
    return 1