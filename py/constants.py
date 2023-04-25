####################################
#  Physical and other constants    #
####################################

import numpy as np

# Units conversions
eV_per_kg = 1.782622e-36
inv_eVs = 6.582119e-16
mP_in_eV = 1.220890e19*1e9
mPred_in_eV = mP_in_eV/np.sqrt(8*np.pi)
GNetwon_in_eV = 1/mP_in_eV**2

# Astro constants
Msol_in_kg = 1.99841e30
Msol_in_eV = Msol_in_kg/eV_per_kg
mP2_in_eVMsol = mP_in_eV*(mP_in_eV/Msol_in_eV)
GNewton = 1/mP2_in_eVMsol
tEddington_in_yr = 4e8
tSalpeter_in_yr = 4.5e7
tSR_in_yr = 4.5e6
yr_in_s = 365.25*24*60*60
inv_tSalpeter = inv_eVs / (yr_in_s*tSalpeter_in_yr) # in eV
inv_tSR = inv_eVs / (yr_in_s*tSR_in_yr) # in eV
