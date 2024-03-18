#####################################
#  Utility functions for statistics #
#####################################

import numpy as np

from numba import njit
from py.bhsr import GammaSR_nlm_bxzh
from py.self_interactions import GammaSR_nlm_eq, is_sr_mode_min, not_bosenova_is_problem_min
from py.constants import *
from py.kerr_bh import alpha

@njit
def cdf_1d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   cdf = []
   cdf_sum = 0
   if data.ndim == 1:
      avals, weights = np.sort(data), np.array(len(data)*[1], dtype='i')
   else:
      indices = np.argsort(data[:,0])
      dat = data[indices]
      avals, weights = dat[:,0], dat[:,1]
   for w in weights:
      cdf_sum += w
      cdf.append(cdf_sum)
   cdf = np.array(cdf, dtype='f')
   cdf /= cdf[-1]
   return avals, cdf

def p_mc_int_no_f(mu: float, samples: np.ndarray[(any,2), float], states: list[tuple[int,int,int]] = [(2,1,1)], tbh: float = tSR_in_yr, sr_function: callable = GammaSR_nlm_bxzh):
   inv_tbh = inv_eVyr/tbh
   n_samples = len(samples[:,0])
   p = n_samples
   for mbh, astar in samples:
      for s in states:
         n, l, m = s
         alph = alpha(mu, mbh)
         # Check if SR mode
         if alph/l <= 0.5:
            # Only now (for efficiency) compute BHSR rate and check if it is fast enough
            srr = sr_function(mu, mbh, astar, n, l, m)
            if srr > inv_tbh:
               p -= 1
               break
   return p/float(n_samples)

def p_mc_int_eq(mu: float, invf: float, samples: np.ndarray[(any,2), float], states: list[tuple[int,int,int]] = [(2,1,1)], tbh: float = tSR_in_yr, sr_function: callable = GammaSR_nlm_bxzh):
   if states != [(2,1,1)]:
      raise ValueError("Only the |nlm> = |211> level is currently supported.")
   inv_tbh = inv_eVyr/tbh
   n_samples = len(samples[:,0])
   p = n_samples
   for mbh, astar in samples:
      for s in states:
         n, l, m = s
         alph = alpha(mu, mbh)
         # Check if SR mode
         if alph/l <= 0.5:
            # Only now (for efficiency) compute BHSR rate and check if it is fast enough
            srr, srr0 = GammaSR_nlm_eq(mu, mbh, astar, invf, n, l, m, sr_function)
            # Check if we reach the equilibrium regime, and the equilibrium rate is fast enough
            if (srr > inv_tbh) and (srr0 > inv_tbh):
               p -= 1
               break
   return p/float(n_samples)

def mc_integration_bosenova(mu: float, invf: float, samples: np.ndarray[(any,2), float], states: list[tuple[int,int,int]] = [(2,1,1)], tbh: float = tSR_in_yr, sr_function: callable = GammaSR_nlm_bxzh):
   n_samples = len(samples[:,0])
   p = n_samples
   for mbh, astar in samples:
      for s in states:
         n, l, m = s
         alph = alpha(mu, mbh)
         # Check if SR mode
         if alph/l <= 0.5:
            # Only now (for efficiency) compute BHSR rate and check if it is fast enough
            srr = sr_function(mu, mbh, astar, n, l, m)
            check1 = is_sr_mode_min(mbh, tbh, srr)
            check2 = not_bosenova_is_problem_min(mu, invf, mbh, tbh, n, srr)
            if check1 and check2:
               p -= 1
               break
   return p/float(n_samples)