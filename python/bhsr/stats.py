#####################################
#  Utility functions for statistics #
#####################################

import fastkde
import numpy as np

from numba import njit
from .bhsr import GammaSR_nlm_bxzh
from .self_interactions import can_grow_max_cloud, GammaSR_nlm_eq, not_bosenova_is_problem
from .constants import *
from .kerr_bh import alpha

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
            check1 = can_grow_max_cloud(mbh, tbh, srr)
            check2 = not_bosenova_is_problem(mu, invf, mbh, tbh, n, srr)
            if check1 and check2:
               p -= 1
               break
   return p/float(n_samples)

@njit
def simple_grid_and_hpd(samples: np.ndarray, xlims: np.ndarray[float], ylims: np.ndarray[float], nbins_post: int = 50, nbins_grid: int = 200, thresh: float = 0.05):
   pdens = []
   nsamples = len(samples[:,0])
   dx = (xlims[-1] - xlims[0])/nbins_post
   dy = (ylims[-1] - ylims[0])/nbins_post
   for i in range(nbins_post):
      tmp = []
      for j in range(nbins_post):
         sel1 = (xlims[0] + i*dx < samples[:,0]) & (samples[:,0] < xlims[0] + (i+1)*dx)
         sel2 = (ylims[0] + j*dy < samples[:,1]) & (samples[:,1] < ylims[0] + (j+1)*dy)
         sel = sel1 & sel2
         tmp.append(np.sum(sel))
      pdens.append(tmp)
   pdens = np.array(pdens)/nsamples
   pdens_sorted = np.sort(pdens.flatten())
   psum, pthresh = 0, 0
   for i in range(len(pdens_sorted)):
      pthresh = pdens_sorted[i]
      psum += pdens_sorted[i]
      if psum > thresh:
         break
   x0 = [xlims[0] + (i+0.5)*dx for i in range(nbins_post) for _ in range(nbins_post)]
   y0 = [ylims[0] + (j+0.5)*dy for _ in range(nbins_post) for j in range(nbins_post)]
   xi = np.linspace(xlims[0], xlims[1], nbins_grid)
   yi = np.linspace(ylims[0], ylims[1], nbins_grid)
   return x0, y0, pdens.flatten(), xi, yi, pthresh

def simple_grid_and_hpd_fastkde(samples: np.ndarray, xlims: np.ndarray[float], ylims: np.ndarray[float], nbins_grid: int = -1, thresh: float = 0.05):
   npts = nbins_grid if nbins_grid > 0 else None
   kde = fastkde.pdf(samples[:,0], samples[:,1], num_points=npts).data
   psum, pthresh = 0, 0
   kde_sorted = np.sort(kde.flatten())
   for i in range(len(kde_sorted)):
      pthresh = kde_sorted[i]
      psum += kde_sorted[i]
      if psum > thresh:
         break
   nyk, nxk = kde.shape
   print("INFO", kde.shape)
   xi = np.linspace(xlims[0], xlims[1], nxk)
   yi = np.linspace(ylims[0], ylims[1], nyk)
   return xi, yi, kde, pthresh
