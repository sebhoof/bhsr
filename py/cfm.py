#####################################################################
#  Computations of the BHSR rates via the continued fraction method #
#####################################################################

from numba import complex128, float64, int16, jit, optional, uint32
import numpy as np

from iminuit import Minuit
from qnm.angular import sep_consts
from scipy.optimize import differential_evolution, root, root_scalar
from .constants import *
from .kerr_bh import omH, rg
from .bhsr import *


# Approximation for Alm eigenvalues (see doi:10.1088/0264-9381/6/7/012)
@jit(float64(int16, int16, int16), nopython=True)
def h_seidel(l: int, m: int = 1, s: int = 0) -> float:
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

@jit(float64(int16, int16, int16), nopython=True)
def flm_seidel_2(l: int, m: int = 1, s: int = 0) -> float:
   return h_seidel(l+1, m, s) - h_seidel(l, m, s) - 1

@jit(float64(int16, int16, int16), nopython=True)
def flm_seidel_4(l: int, m: int = 1, s: int = 0) -> float:
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

@jit(nopython=True, cache=True)
def alm_approx(c: complex , l: int, m: int = 1, s: int = 0) -> complex:
   fvals = [l*(l+1), flm_seidel_2(l, m, s), flm_seidel_4(l, m, s)]
   expansion = [f*pow(c,2*i) for i,f in enumerate(fvals)]
   return sum(expansion)

# Compute continued fraction based on arXiv:0705.2880

@jit(float64(float64), nopython=True)
def mbh_in_eV(mbh: float) -> float:
   return mbh*Msol_in_eV

# Angular eigenvalues for the Teukolsky equation, Eq. (46)
# Computed using the 'qnm' package
def angular_ev(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int) -> complex:
   c = rg(mbh)*astar*np.sqrt(omega*omega - mu*mu)
   if np.abs(c) > 3:
      print("WARNING. |c| > 3 detected. Use qlm.")
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

@jit(complex128(uint32, complex128), nopython=True)
def alpha_n(n: int, c0: complex) -> complex:
   return n*n + (c0 + 1)*n + c0

@jit(complex128(uint32, complex128, complex128), nopython=True)
def beta_n(n: int, c1: complex, c3: complex) -> complex:
   return -2*n*n + (c1 + 2)*n + c3

@jit(complex128(uint32, complex128, complex128), nopython=True)
def gamma_n(n: int, c2: complex, c4: complex) -> complex:
   return n*n + (c2 - 3)*n + c4

def continued_fraction(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int) -> [float, float]:
   # Simple method for now: set the residual terms to zero
   # For alternatives, see Sec. II C in https://arxiv.org/pdf/1410.7698.pdf
   c0, c1, c2, c3, c4, cN1, cN2 = cfunctions(omega, mbh, astar, mu, l, m)
   nmax = 1000
   fr = (-1+0j) + cN1/np.sqrt(nmax) + cN2/nmax
   # fr = 0+0j
   fr0 = beta_n(0, c1, c3)/alpha_n(0, c0)
   for i in np.flip(range(1,nmax+1)):
      alph = alpha_n(i, c0)
      beta = beta_n(i, c1, c3)
      gam = gamma_n(i, c2, c4)
      fr = gam/(beta - alph*fr)
   return fr0 - fr
   # return fr/beta_0 - (1+0j)
   # return np.log(beta_0/fr)
   # return np.array([beta_0.real/fr.real - 1.0, beta_0.imag/fr.imag - 1.0])


def root_equation(x: [float, float], mbh: float, astar: float, mu: float, l: int, m: int) -> float:
   omega = x[0] + 1j*x[1]
   z = continued_fraction(omega, mbh, astar, mu, l, m)
   return z

def min_equation(x: [float, float], mbh: float, astar: float, mu: float, l: int, m: int) -> float:
   z = root_equation(x, mbh, astar, mu, l, m)
   return 2*np.log(np.abs(z))
   # return -1/np.abs(z)**
   # return np.abs(z)**2

def find_root(mbh: float, astar: float, mu: float, n: int = 2, l: int = 1, m: int = 1, verbose: bool = False) -> complex:
   omR = omegaHyperfine(mu, mbh, astar, n, l, m)
   # omI = GammaSR_nlm_nr(mu, mbh, astar, n, l, m)
   _, omI = omega_nlm_bxzh(mu, mbh, astar, 2, 1, 1)
   om_max = m*omH(mbh, astar)
   # lgmu = np.log10(mu)
   if omR > 0 and omI > 0:
      lgomR = np.log10(omR)
      lgomI = np.log10(omI)
      foo = lambda x0, x1: min_equation([x0, pow(10, x1)], mbh, astar, mu, l, m)
      res = Minuit(foo, x0=omR, x1=lgomI-0.1)
      # foo = lambda x0, x1: min_equation([x0, x1], mbh, astar, mu, l, m)
      # res = Minuit(foo, x0=0.9*omR, x1=0.1*omI)
      # res.limits["x0"] = (0.8*omR, min(mu, 1.1*omR))
      res.limits["x0"] = (0.925*omR, min(1.025*omR, om_max))
      res.limits["x1"] = (lgomI-0.5, min(lgomI+0.5, lgomR))
      res.tol = 0
      res.migrad()
      res.simplex()
      om = res.values["x0"] + 1j*pow(10, res.values["x1"])
      """
      # om = res.values["x0"] + 1j*res.values["x1"]
      foo = lambda x: min_equation([x[0], pow(10, x[1])], mbh, astar, mu, l, m)
      res = differential_evolution(foo, bounds=[[0.9*omR, min(mu, 1.1*omR)], [lgomI-2, lgomI+2]], x0=[0.95*mu, lgomI], strategy='rand1exp', maxiter=1000, popsize=50, tol=1e-4, atol=1e-4, mutation=(0.5, 1), recombination=0.7, polish=True, init='sobol')
      om = res.x[0] + 1j*pow(10, res.x[1])
      def foo(x):
         z = root_equation([x[0], pow(10, x[1])], mbh, astar, mu, l, m)
         return [z.real, z.imag]
      # res = root(foo, x0=[omR, lgomI], method='anderson', options = { 'ftol': 1e-7, 'fatol': 1e-7 })
      res = root(foo, x0=[omR, lgomI], tol=1e-10, method='broyden1')
      # res = root(foo, x0=[omR, lgomI], method='broyden1', options = { 'ftol': 1e-9, 'fatol': 1e-9 })
      """
      """
      foo = lambda x: root_equation([x, omI], mbh, astar, mu, l, m).real
      res = root_scalar(foo, method='brentq', bracket=[0.8*omR, omR])
      x0 = res.root
      foo = lambda x: root_equation([x0, pow(10, x)], mbh, astar, mu, l, m).imag
      res = root_scalar(foo, method='brentq', x0=lgomI, bracket=[lgomI-3, min(lgomI+3, np.log10(mu))])
      x1 = pow(10, res.root)
      om = x0 + 1j*x1
      """
   else:
      # foo = lambda x0, x1: min_equation([x0, x1], mbh, astar, mu, l, m)
      # res = Minuit(foo, x0=omR, x1=omI)
      # res.limits["x0"] = (0.8*omR, min(mu, 1.25*omR))
      # res.limits["x1"] = (-mu, 0)  
      # # res.tol = 0
      # res.migrad()
      # om = res.values["x0"] + 1j*res.values["x1"]
      return omR + 1j*omI
   if verbose:
       print(om)
   return om