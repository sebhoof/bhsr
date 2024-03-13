###############################################################
#  Functions to consider the effects of ULB self-interactions #
###############################################################

import numpy as np

from .constants import *
from .bhsr import *

# Constant obtained from numerical simulations for computing n_bose()
# Ref.: https://arxiv.org/pdf/1411.2263.pdf
c0_n_bose = 5.0

@njit
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
        - Ref.: Eq. (9) in https://arxiv.org/pdf/1411.2263.pdf
    """
    alph = alpha(mu, mbh)
    x = n*n*(mbh/10)*1.0/(invf*mPred_in_GeV)
    return 1e78*c0_n_bose*x*x/alph**3

@njit
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
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = sr_rate > inv_t*np.log(nb)*(nm/nb)
    if np.isnan(res):
        res = 0
    return res

# TODO: Duplicate function; remove/consolidate.
@njit("boolean(float64, float64, float64, float64, uint8, float64)")
def not_bosenova_is_problem_min(mu: float, invf: float, mbh: float, tbh: float, n: int, min_sr_rate: float) -> bool:
    nm = n_max(mbh)
    nb = n_bose(mu, invf, mbh, n)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = min_sr_rate > inv_t*np.log(nb)*(nm/nb)
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
    mbh_p, mbh_m = mbh+sigma_level*mbh_err, max(0,mbh-sigma_level*mbh_err)
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
                sr_checks.append(is_sr_mode_min(mm, tbh, srr)*not_bosenova_is_problem_min(mu, invf, mm, tbh, n, srr))
        if sum(sr_checks) == 0:
            return 0
    return 1