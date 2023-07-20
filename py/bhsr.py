###############
#  BHSR rates #
###############

import numpy as np

from math import factorial, prod
from fractions import Fraction
from .constants import mP_in_GeV, inv_eVs, yr_in_s
from .kerr_bh import *

# BHSR rates using Dettweiler's approx
# Eqs. 5, 13, 14, 15 in https://arxiv.org/pdf/2009.07206.pdf
# and further input from https://arxiv.org/pdf/1501.06570.pdf and https://arxiv.org/pdf/1411.2263.pdf
# N.B. Difference to Eq. 14 in https://arxiv.org/pdf/1805.02016.pdf

def omega(mu, mbh, n, l, m):
    x = alpha(mu, mbh)/n
    return mu*(1.0 - 0.5*x*x)

def c_nl(n, l):
    x = Fraction(factorial(n+l) * 2**(4*l+1), factorial(n-l-1) * n**(2*l+4) )
    y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
    return x*y*y

def c_nl_float(n, l):
    c_nl_fr = c_nl(n, l)
    return 1.0*c_nl_fr

#def x_mn(mu, astar, n, l, m):
#    factors = [k*k*(1.0-astar*astar) + 4*r_plus*r_plus*(m*omega(n,l,m) - mu*mu)**2 for k in range(1,l+1)]
#    return prod(factors)

#def GammaSR_nlm(mu, mbh, astar, n, l, m):
#    x = 2*mu*r_plus(mbh, astar)*(m*omH(mbh, astar) - mu)
#    y = alpha(mu, mbh)**(4*l+4)
#    return x*y*a_nl(n, l)*x_mn(mu, astar, n, l, m)

def GammaSR_nlm_mod(ma, mbh, astar, n, l, m):
    al = alpha(ma, mbh)
    marp = al*(1 + np.sqrt(1-astar*astar))
    x = m*astar - 2*marp
    y = al**(4*(l+1))
    factors = [(k*k)*(1.0-astar*astar) + x**2 for k in range(1,l+1)]
    return c_nl_float(n, l) * ma*x * prod(factors) * al**(4*(l+1))


# Functions relating to the "non-relativistic approximation"
# see e.g. Fig. 5 in https://arxiv.org/pdf/1004.3558.pdf
def c_nl_nr(n, l):
    x = Fraction(factorial(2*l+n+1) * 2**(4*l+2), factorial(n) * (l+n+1)**(2*l+4) )
    y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
    return x*y*y

def c_nl_nr_float(n, l):
    c_nl_fr = c_nl_nr(n, l)
    return 1.0*c_nl_fr

def GammaSR_nlm_nr(ma, mbh, astar, n, l, m):
    al = alpha(ma, mbh)
    marp = al*(1 + np.sqrt(1-astar*astar))
    x = 2.0*(0.5*m*astar - marp)
    factors = [(k*k)*(1.0-astar*astar) + x*x for k in range(1,l+1)]
    return  ma*x * al**(4*(l+1)) * c_nl_nr_float(n, l) * prod(factors)

# Compute spindown rate according to quasi-equilibrium approximation
# Follow O. Simon's unpublished notes
def GammaSR_322xBH_211x211(ma, mbh, astar, fa):
    al = alpha(ma, mbh)
    return 4.3e-7 * ma*(1.0 + np.sqrt(1.0 - astar*astar))*pow(al, 11)*pow(mP_in_GeV/fa, 4)

def GammaSR_211xinf_322x322(ma, mbh, fa):
    al = alpha(ma, mbh)
    return 1.1e-8 * ma*pow(al, 8)*pow(mP_in_GeV/fa, 4)

def n_eq_211(ma, mbh, astar, fa):
    sr0 = GammaSR_nlm_mod(ma, mbh, astar, 2, 1, 1)
    sr_3b22 = GammaSR_322xBH_211x211(ma, mbh, astar, fa)
    sr_2i33 = GammaSR_211xinf_322x322(ma, mbh, fa)
    return 2.0*np.sqrt(sr0*sr_2i33/3.0)/sr_3b22

def GammaSR_nlm_eq(ma, mbh, astar, fa):
    sr0 = GammaSR_nlm_mod(ma, mbh, astar, 2, 1, 1)
    neq = n_eq_211(ma, mbh, astar, fa)
    return neq*sr0

def n_max(mbh):
    da0 = 0.1
    return 1e76 * (da0/0.1) * (mbh/10)**2

def is_sr_mode(mu, mbh, astar, tbh, n, l, m):
    nm = n_max(mbh)
    inv_t = inv_eVs / (yr_in_s*tbh)
    res = GammaSR_nlm_mod(mu, mbh, astar, n, l, m) > inv_t*np.log(nm)
    if np.isnan(res):
        res = 0
    return res

def is_sr_mode_min(mu, min_sr_rate, mbh, tbh):
    nm = n_max(mbh)
    inv_t = inv_eVs / (yr_in_s*tbh)
    return min_sr_rate > inv_t*np.log(nm)