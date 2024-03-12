#####################################################################
#  Computations of the BHSR rates via the continued fraction method #
#####################################################################

from numba import njit
import numpy as np

from iminuit import Minuit
from qnm.angular import sep_consts
# from scipy.optimize import differential_evolution, root, root_scalar
from scipy.optimize import minimize_scalar
from .constants import *
from .kerr_bh import omH, rg
from .bhsr import *


# Approximation for the eigenvalues of the spin-weighted spheroidal(!) functions

@njit("float64(uint8, int16, uint8)")
def h_seidel(l: int, m: int = 1, s: int = 0) -> float:
   """
   Helper function for alm_approx.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$h(l)\f$ function.

   Notes:
      - Eq. (8) in https://doi.org/10.1088/0264-9381/6/7/012
   """
   num = l*(l*l - m*m)
   denom = 2*(l-0.5)*(l+0.5)
   if s > 0:
      mabs = np.abs(m)
      s1 = max(mabs, s)
      s2 = m*s/max(mabs, s)
      l2 = l*l
      num = (l2 - s1*s1)*(l2 - s*s)*(l2 - s2*s2)
      denom *= l2*l
   return num/denom

@njit("float64(uint8, int16, uint8)")
def flm_seidel_2(l: int, m: int = 1, s: int = 0) -> float:
   """
   Helper function for alm_approx.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$_sf_2^{lm}\f$ contribution.

   Notes:
      - Eq. (10c) in https://doi.org/10.1088/0264-9381/6/7/012
   """
   return h_seidel(l+1, m, s) - h_seidel(l, m, s) - 1

@njit("float64(uint8, int16, uint8)")
def flm_seidel_4(l: int, m: int = 1, s: int = 0) -> float:
   """
   Helper function for alm_approx.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$_sf_4^{lm}\f$ contribution.

   Notes:
      - Eq. (10e) in https://doi.org/10.1088/0264-9381/6/7/012
   """
   hl = h_seidel(l, m, s)
   hlp1 = h_seidel(l+1, m, s)
   hlp2 = h_seidel(l+2, m, s)
   twol = 2*l
   l2 = l*l
   lm1 = l-1
   lp1 = l+1
   lp2 = l+2
   res = (hlp1 - lp2*hlp2/(twol+3))*hlp1/(2*lp1)
   res += (hlp1/lp1 - hl)*hl/twol
   res += lm1*h_seidel(l-1, m, s)*hl/(twol*(twol-1))
   if s > 0:
      lp2sq = lp2*lp2
      lm1sq = lm1*lm1
      lp1sq = lp1*lp1
      res += 4*( hlp1/(lp1sq*lp2sq) - hl/(l2*lm1sq))*m*m*pow(s,4)/(l2*lp1sq)
   return res

@njit("complex128(complex128, uint8, int16, uint8)")
def alm_approx(c: complex , l: int, m: int = 1, s: int = 0) -> complex:
   """
   The eigenvalues \f$A_{lm}\f$ of the spin-weighted spheroidal(!) functions.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$A_{lm}\f$ eigenvalues.

   Notes:
      - Eq. (7) in https://doi.org/10.1088/0264-9381/6/7/012 with \f$A_{lm} \equiv _sE_l^m - s(s+1)\f$.
   """
   fvals = [l*(l+1), flm_seidel_2(l, m, s), flm_seidel_4(l, m, s)]
   expansion = [f*pow(c,2*i) for i,f in enumerate(fvals)]
   return sum(expansion)

# Compute continued fraction based on arXiv:0705.2880

# Angular eigenvalues for the Teukolsky equation, Eq. (46)
def angular_ev(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int) -> complex:
   c = rg(mbh)*astar*np.sqrt(omega*omega - mu*mu)
   if np.abs(c) > 3:
      # Need to use the 'qnm' package here
      # print("WARNING. |c| > 3 detected. Use qlm.")
      return np.sort(sep_consts(s=0, c=c, m=m, l_max=l))[-1]
   return alm_approx(c, l, m, s=0)

# Auxiliary functions in Eqs (40)--(44)
def cfunctions(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int) -> tuple:
   alm = angular_ev(omega, mbh, astar, mu, l, m)
   a = astar
   a2 = a*a
   b = np.sqrt(1 - a2)
   om = rg(mbh)*omega
   om2 = om*om
   mu_r = rg(mbh)*mu
   mu2 = mu_r*mu_r
   # Choose appropriate sign for bound states
   q = np.sqrt(mu2 - om2)
   # cN2 = 0.75 + (2*(b+1)*om2 - (2*b+1)*mu2)/q
   q = -np.sign(q.real)*q
   cN1 = 4*b*q
   cN2 = 0.75 + (2*(b+1)*om2 - (2*b+1)*mu2)/q
   q2 = q*q
   x = (om - 0.5*m*a)/b
   y = 2j*(om + x)
   z = om - 1j*q
   z2 = z*z/q
   # Compute the numerical coefficients
   c0 = 1 - y
   c1 = -4 + 2*y + 4*(b + 1)*q - 2*(q2 + om2)/q
   c2 = 3 - y - 2*(q2 - om2)/q
   c3 = 2j*z*z2 + a2*q2 + 2j*m*a*q + (z2 + 1)*(2j*x + 2*b*q - 1) - alm
   c4 = z2*z2 + 2j*z2*(om - x)
   return c0, c1, c2, c3, c4, cN1, cN2

# Auxiliary functions in Eqs (37)--(39)

@njit("complex128(uint8, complex128)")
def alpha_n(n: int, c0: complex) -> complex:
   return n*n + (c0 + 1)*n + c0

@njit("complex128(uint8, complex128, complex128)")
def beta_n(n: int, c1: complex, c3: complex) -> complex:
   return -2*n*n + (c1 + 2)*n + c3

@njit("complex128(uint8, complex128, complex128)")
def gamma_n(n: int, c2: complex, c4: complex) -> complex:
   return n*n + (c2 - 3)*n + c4

def continued_fraction(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int, nmax: int = 2000) -> [float, float]:
   # Set the residual terms to zero
   # For alternatives, see Sec. II C in https://arxiv.org/pdf/1410.7698.pdf
   c0, c1, c2, c3, c4, cN1, cN2 = cfunctions(omega, mbh, astar, mu, l, m)
   fr = (-1+0j) + cN1/np.sqrt(nmax) + cN2/nmax # Improved residual term
   # fr = 0+0j
   fr0 = beta_n(0, c1, c3)/alpha_n(0, c0)
   for i in np.flip(range(1,nmax+1)):
      alph = alpha_n(i, c0)
      beta = beta_n(i, c1, c3)
      gam = gamma_n(i, c2, c4)
      fr = gam/(beta - alph*fr)
   # return fr0 - fr
   return fr0/fr - 1
   # return fr/fr0
   # return fr/beta_0 - (1+0j)
   # return np.log(beta_0/fr)
   # return np.array([beta_0.real/fr.real - 1.0, beta_0.imag/fr.imag - 1.0])

def root_equation(x: [float, float], mbh: float, astar: float, mu: float, l: int, m: int) -> float:
   omega = x[0] + 1j*x[1]
   z = continued_fraction(omega, mbh, astar, mu, l, m)
   return np.log(z+1)

def min_equation(x: [float, float], mbh: float, astar: float, mu: float, l: int, m: int) -> float:
   z = root_equation(x, mbh, astar, mu, l, m)
   # return np.abs(z)
   # return -1/np.abs(z)**2
   return np.abs(z)
   # z = root_equation(x, mbh, astar, mu, l, m)
   # return z.imag*z.imag + (z.real - 1)**2
   # return np.nanmax([-1e10, 2.0*np.log(np.abs(z))])


def find_root(mbh: float, astar: float, mu: float, n: int = 2, l: int = 1, m: int = 1, verbose: bool = False) -> complex:
   alph = alpha(mu, mbh)
   omR = omegaHyperfine(mu, mbh, astar, n, l, m)
   # omI = GammaSR_nlm_nr(mu, mbh, astar, n, l, m)
   # _, omI = omega_nlm_bxzh(mu, mbh, astar, 2, 1, 1)
   _, omI = omega_nlm_bxzh(mu, mbh, astar, n, l, m)
   #om_max = m*omH(mbh, astar)
   # lgmu = np.log10(mu)
   if omR > 0 and omI > 0 and alph > 0:
      # lgomR = np.log10(omR)
      # lgomI = np.log10(omI)
      # foo = lambda x0, x1: min_equation([x0, pow(10, x1)], mbh, astar, mu, l, m)
      """
      foo = lambda oR, oI: min_equation([oR, oI], mbh, astar, mu, l, m)
      mnt = Minuit(foo, oR=omR, oI=1.05*omI)
      # foo = lambda x0, x1: min_equation([x0, x1], mbh, astar, mu, l, m)
      # res = Minuit(foo, x0=0.9*omR, x1=0.1*omI)
      # mnt.limits["oR"] = (0.995*omR, min(1.005*omR, om_max))
      alph = alpha(mu, mbh)
      factor = 0.5*alph*alph
      om_lo = mu*(1 - 0.5*factor*( 1.0/((n-1)*(n-1)) + 1.0/(n*n) ))
      om_hi = min(mu*(1 - factor/(n*n)), om_max)
      mnt.limits["oR"] = (om_lo, om_hi)
      # Values informed from Figs 2, 3 of https://arxiv.org/pdf/2201.10941.pdf
      mnt.limits["oI"] = (0.5*omI, min(3*omI, omR))
      mnt.tol = 1e-10
      mnt.fixed["oI"] = True
      mnt.migrad()
      mnt.fixed["oR"] = True
      mnt.fixed["oI"] = False
      mnt.migrad()
      om = mnt.values["oR"] + 1j*mnt.values["oI"]
      """;
      cost_oR = lambda x: np.log(np.abs(root_equation([x, omI], mbh, astar, mu, l, m)))
      mR = Minuit(cost_oR, x=omR)
      mR.tol = 1e-10
      factor = 0.5*alph*alph
      om0 = mu*(1 - 0.5*factor*( 1.0/((n-1)*(n-1)) + 1.0/(n*n) ))
      om1 = mu*(1 - factor/(n*n))
      mR.limits["x"] = (om0, om1)
      mR.migrad()
      cost = lambda x, lgy: np.abs(root_equation([x, 10**lgy], mbh, astar, mu, l, m))
      mRI = Minuit(cost, x=mR.values["x"], lgy=np.log10(omI))
      mRI.tol = 1e-10
      mRI.limits["x"] = (om0, om1)
      mRI.limits["lgy"] = (np.log10(0.7*omI), min(np.log10(10*omI), np.log10(0.1*omR)))
      mRI.migrad()
      om = mRI.values["x"] + 1j*pow(10, mRI.values["lgy"])
      """
      factor = 0.5*alph*alph
      om0 = mu*(1 - 0.5*factor*( 1.0/((n-1)*(n-1)) + 1.0/(n*n) ))
      om1 = omR
      om2 = mu*(1 - factor/(n*n))
      #if om_guess > om_hi:
      #   om_guess = om_hi
      #   om_hi = mu*(1 - 0.5*factor*( 1.0/(n*n) + 1.0/((n+1)*(n+1)) ))
      foo = lambda x: min_equation([x, omI], mbh, astar, mu, l, m)
      try:
         res = minimize_scalar(foo, bracket=(om0, om1, om2), method='Brent')
         # res = minimize_scalar(foo, bounds=(om0, om2), method='Bounded')
         om = res.x
      except:
         print(f"Error! mu = {mu:.3e}, alpha = {alph:.3e} | om = ({om0:.3e}, {om1:.3e}, {om2:.3e}), f(z) = ({foo(om0):.3e}, {foo(om1):.3e}, {foo(om2):.3e})")
         om = omR
      foo = lambda lgx: min_equation([om, pow(10,lgx)], mbh, astar, mu, l, m)
      lgomI = np.log10(omI)
      # foo = lambda x: min_equation([om, x], mbh, astar, mu, l, m)
      try:
         res = minimize_scalar(foo, bracket=(lgomI-1, lgomI, lgomI+0.5), method='Brent')
         # res = minimize_scalar(foo, bounds=(lgomI-0.5, lgomI+1), method='Bounded')
         # res = minimize_scalar(foo, bracket=(0.1*omI, omI, min(2*omI, om)), method='Brent')
         # res = minimize_scalar(foo, bounds=(0.1*omI, omI, min(2*omI, om)), method='Brent')
         om += 1j*pow(10, res.x)
         # om += 1j*res.x
      except:
         # print(f"Error! mu = {mu:.3e}, alpha = {alph:.3e} | lg = ({lgomI-1:.3e}, {lgomI:.3e}, {min(lgomI+1, np.log10(om)):.3e}), f(z) = ({foo(lgomI-1):.3e}, {foo(lgomI):.3e}, {foo(min(lgomI+1, np.log10(om))):.3e})")
         print(f"Error! mu = {mu:.3e}, alpha = {alph:.3e} | gam = ({0.1*omI:.3e}, {omI:.3e}, {min(2*omI, om):.3e}), f(z) = ({foo(0.1*omI):.3e}, {foo(omI):.3e}, {foo(min(2*omI, om)):.3e})")
         om += 1j*omI
      """;
   else:
      if verbose:
         print("Estimates for real or imaginary part of omega are not positive:", om)
      # foo = lambda x0, x1: min_equation([x0, x1], mbh, astar, mu, l, m)
      # res = Minuit(foo, x0=omR, x1=omI)
      # res.limits["x0"] = (0.8*omR, min(mu, 1.25*omR))
      # res.limits["x1"] = (-mu, 0)  
      # # res.tol = 0
      # res.migrad()
      # om = res.values["x0"] + 1j*res.values["x1"]
      return omR + 1j*omI
   return om