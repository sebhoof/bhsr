#####################################
#  Utility functions for statistics #
#####################################

import numpy as np

from numba import njit

@njit
def cdf_1d(data):
   cdf = []
   cdf_sum = 0
   if data.ndim == 1:
      avals, weights = np.sort(data), np.array(len(data)*[1], dtype='int')
   else:
      indices = np.argsort(data[:,0])
      dat = data[indices]
      avals, weights = dat[:,0], dat[:,1]
   for w in weights:
      cdf_sum += w
      cdf.append(cdf_sum)
   cdf = np.array(cdf, dtype='float')
   cdf /= cdf[-1]
   return avals, cdf