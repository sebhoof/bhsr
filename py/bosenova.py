#######################
#  Bosenova functions #
#######################

import numpy as np

from .constants import *
from .bhsr import *

def n_bose(mu, invf, mbh, n=2):
    c0 = 5
    f = 1.0/invf
    alph = alpha(mu, mbh)
    return 1e78*c0* n**4 * (mbh/10)**2 * (f*1e9/mPred_in_eV)**2 / alph**3

def bosenova_f0(mu, mbh, n=2):
    c0 = 5
    da0 = 0.1
    f0 = 2e16 * (alpha(mu, mbh)/(0.4*n))**(3.0/2.0) * np.sqrt(da0/0.1) * np.sqrt(5/c0) / np.sqrt(n)
    return f0

def bosenova_check_old(invf, mu, mbh, n=2):
    return 1.0/invf > bosenova_f0(mu, mbh, n)

def not_bosenova_is_problem(mu, invf, mbh, a, tbh, n, l, m):
    nm = n_max(mbh)
    nb = n_bose(mu, invf, mbh, n)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = GammaSR_nlm_mod(mu, mbh, a, n, l, m) > inv_t*np.log(nb)*(nm/nb)
    if np.isnan(res):
        res = 0
    return res

def not_bosenova_is_problem_min(mu, min_sr_rate, invf, mbh, tbh, n):
    nm = n_max(mbh)
    nb = n_bose(mu, invf, mbh, n)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = min_sr_rate > inv_t*np.log(nb)*(nm/nb)
    return res