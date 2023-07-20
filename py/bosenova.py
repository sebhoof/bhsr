###############################
#  Bosenova-related functions #
###############################

import numpy as np

from .constants import *
from .bhsr import *

# Constant obtained from numerical simulations for computing n_bose()
# Ref.: https://arxiv.org/pdf/1411.2263.pdf
c0_n_bose = 5.0

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

#def bosenova_check_old(invf, mu, mbh, n=2):
#    return 1.0/invf > bosenova_f0(mu, mbh, n)

def not_bosenova_is_problem(mu, invf, mbh, a, tbh, n, l, m):
    nm = n_max(mbh)
    nb = n_bose(mu, invf, mbh, n)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = GammaSR_nlm(mu, mbh, a, n, l, m) > inv_t*np.log(nb)*(nm/nb)
    if np.isnan(res):
        res = 0
    return res

def not_bosenova_is_problem_min(mu, min_sr_rate, invf, mbh, tbh, n):
    nm = n_max(mbh)
    nb = n_bose(mu, invf, mbh, n)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = min_sr_rate > inv_t*np.log(nb)*(nm/nb)
    return res