##################
#  BHSR rates    #
##################

import numpy as np
from math import factorial, prod
from fractions import Fraction

from .constants import GNewton

# BHSR rates using Dettweiler's approx
# Eqs. 5, 13, 14, 15 from https://arxiv.org/pdf/2009.07206.pdf
# N.B. Difference to https://arxiv.org/pdf/1805.02016.pdf, Eq. 14
# Also input from 1501.06570 and 1411.2263

def rg(mbh):
    return GNewton*mbh

def alpha(ma, mbh):
    return rg(mbh)*ma

def r_plus(mbh, astar):
    return rg(mbh)*(1 + np.sqrt(1 - astar*astar))

def r_minus(mbh, astar):
    return rg(mbh)*(1 - np.sqrt(1 - astar*astar))

def omH(mbh, astar):
    return 0.5*astar/r_plus(mbh, astar)

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