###############################################################
#  Functions to consider the effects of ULB self-interactions #
###############################################################

import numpy as np

from .constants import *
from .kerr_bh import *
from .bhsr import *

### Routines for bosenovae

c0_n_bose = 5.0

@njit("float64(float64, float64, float64, uint8)")
def n_bose(mu: float, invf: float, mbh: float, n: int = 2) -> float:
    """
    Calculates the number of bosons in a black hole superradiant cloud that triggers a bosenova.

    Parameters:
        mu (float): Boson mass in eV.
        invf (float): Inverse of the boson decay constant in GeV^-1.
        mbh (float): Black hole mass in Msol.
        n (int, optional): Principal quantum number (default: 2).

    Returns:
        float: Number of bosons in the superradiant cloud.

    Notes:
        - Ref.: Eq. (9) in https://arxiv.org/pdf/1411.2263.pdf, derived from https://arxiv.org/pdf/1203.5070.pdf
    """
    alph = alpha(mu, mbh)
    x = n*n*(mbh/10)*1.0/(invf*mPred_in_GeV)
    return 1e78*c0_n_bose*x*x/alph**3

@njit("float64(float64, float64, uint8)")
def bosenova_fcrit(mu: float, mbh: float, n: int = 2) -> float:
    """
    Calculates the critical boson decay constant at which a bosenova in a black hole superradiant cloud
    is expected to happen before (most of the) black hole spin has been depleted.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        n (int, optional): Principal quantum number (default: 2).

    Returns:
        float: Critical boson decay constant in GeV.

    Notes:
        - Computed by equating n_bose() with n_max(da = 0.1)
    """
    da0 = 0.1
    f0 = 2e16 * pow(alpha(mu, mbh)/(0.4*n), 1.5) * np.sqrt( (da0/0.1) * (5.0/c0_n_bose) / n )
    return f0

@njit("boolean(float64, float64, float64, float64, uint8, float64)")
def not_bosenova_is_problem(mu: float, invf: float, mbh: float, tbh: float, n: int, sr_rate: float) -> bool:
    """
    Check if bosenovae do not pose a problem for the parameter constraints.

    Parameters:
        mu (float): Boson mass in eV.
        invf (float): Inverse of the boson decay constant in GeV^-1.
        mbh (float): Black hole mass in Msol.
        tbh (float): Black hole timescale in yr.
        n (int): Principal quantum number.
        sr_rate (float): Superradiance rate in eV.

    Returns:
        bool: True if bosenovae do not pose a problem for the parameter constraints.
    """
    nm = n_max(mbh)
    nb = n_bose(mu, invf, mbh, n)
    inv_tbh = inv_eVyr/tbh
    res = sr_rate > inv_tbh*np.log(nb)*(nm/nb)
    if np.isnan(res):
        res = 0
    return res

def is_box_allowed_bosenova(mu: float, invf: float, bh_data, states: list[tuple[int,int,int]] = [(ell+1, ell, ell) for ell in range(1,6)], sigma_level: float = 2, sr_function: callable = GammaSR_nlm_nr) -> bool:
    """
    Check if a configuration is allowed by superradiance and bosenovae, using the `box method`.

    Parameters:
        mu (float): Boson mass in eV.
        invf (float): Inverse of the boson decay constant in GeV^-1.
        bh_data (tuple): Black hole data (tbh, mbh, mbh_err, a, a_err_p, a_err_m).
        states (list[tuple[int,int,int]]): List of levels \f$|nlm\rangle\f$ (default: all \f$n \leq 5\f$).
        sigma_level (float): Confidence level for the exclusion (default: 2)
        sr_function (callable): Superradiance rate function (default: GammaSR_nlm_nr).

    Returns:
        bool: True if the configuration is allowed by superradiance and bosenovae.
    """
    _, tbh, mbh, mbh_err, a, _, a_err_m = bh_data
    # Coservative approach by choosing the shortest BH time scale
    tbh = min(tEddington_in_yr, tbh)
    mbh_p, mbh_m = mbh+sigma_level*mbh_err, max(0, mbh-sigma_level*mbh_err)
    a_m = max(0, a-sigma_level*a_err_m)
    # A configuration is only excluded if sr_checks = [0, 0]
    for mm in np.linspace(mbh_m, mbh_p, 25):
        sr_checks = []
        for s in states:
            n, l, m = s
            # Check SR condition
            alph = alpha(mu, mm)
            if alph/l <= 0.5:
                srr = sr_function(mu, mm, a_m, n, l, m)
                sr_checks.append(can_grow_max_cloud(mm, tbh, srr)*not_bosenova_is_problem(mu, invf, mm, tbh, n, srr))
        if sum(sr_checks) == 0:
            return 0
    return 1


### Routines for the equilibrium regime

@njit
def GammaSR_322xBH_211x211(mu: float, mbh: float, astar: float, invf: float) -> float:
    """
    Calculates the rate factor that corresponds to the damping rate of forced ocillations in the boson cloud,
    from the interacting |211> and |322> states and the BH.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter.
        invf (float): Inverse boson decay constant in GeV^-1.

    Returns:
        float: The rate factor in eV.

    Notes:
        - Ref.: Table I, https://arxiv.org/pdf/2011.11646.pdf
    """
    al = alpha(mu, mbh)
    return 4.3e-7 * mu*(1.0 + np.sqrt(1.0-astar*astar))*pow(al, 11)*pow(mP_in_GeV*invf, 4)

@njit
def GammaSR_211xinf_322x322(mu: float, mbh: float, invf: float) -> float:
    """
    Calculates the rate factor that corresponds to scalar emissions from the interacting |211> and |322> states
    to infinity.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        invf (float): Inverse boson decay constant in GeV^-1.

    Returns:
        float: The rate factor in eV.

    Notes:
        - Ref.: Table I, https://arxiv.org/pdf/2011.11646.pdf
    """
    al = alpha(mu, mbh)
    return 1.1e-8 * mu*pow(al, 8)*pow(mP_in_GeV*invf, 4)

@njit
def n_eq_211(mu: float, mbh: float, astar: float, invf: float, sr_0_211: float) -> float:
    """
    Calculates the equilibrium occupation number of the |211> state in a self-interacting boson cloud.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter.
        invf (float): Inverse boson decay constant in GeV^-1.
        sr_0_211 (float): Superradiance rate for the |211> level.

    Returns:
        float: The equilibrium occupation number.

    Notes:
        - Ref.: Eq. (55a) in https://arxiv.org/pdf/2011.11646.pdf
    """
    sr_3b22 = GammaSR_322xBH_211x211(mu, mbh, astar, invf)
    sr_2i33 = GammaSR_211xinf_322x322(mu, mbh, invf)
    return 2*np.sqrt(sr_0_211*sr_2i33/3.0)/sr_3b22

def GammaSR_nlm_eq(mu: float, mbh: float, astar: float, invf: float, n: int = 2, l: int = 1, m: int = 1, sr_function: callable = GammaSR_nlm_bxzh) -> float:
    """
    Calculates the effective SR rate of the |211> state in a self-interacting boson cloud.

    Parameters:
        mu (float): Boson mass in eV.
        mbh (float): Black hole mass in Msol.
        astar (float): Dimensionless black hole spin parameter.
        invf (float): Inverse of the boson decay constant in GeV.
        n (int): Principal quantum number (currently only n = 2).
        l (int): Orbital angular momentum quantum number (currently only l = 1).
        m (int): Magnetic quantum number (currently only m = 1).

    Returns:
        float: The equilibrium occupation number.
    """
    if n != 2 or l != 1 or m != 1:
        raise ValueError("Only the |nlm> = |211> level is currently supported by GammaSR_nlm_eq.")
    sr0 = sr_function(mu, mbh, astar, n, l, m)
    neq = n_eq_211(mu, mbh, astar, invf, sr0)
    return neq*sr0, sr0

def is_box_allowed_211(mu: float, invf: float, bh_data: list[int], sr_function: callable, sigma_level: float = 2):
    _, tbh, mbh, mbh_err, a, _, a_err_m = bh_data
    # Coservative approach by choosing the shortest BH time scale
    tbh = min(tEddington_in_yr, tbh)
    mbh_p, mbh_m = mbh+sigma_level*mbh_err, max(0, mbh-sigma_level*mbh_err)
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