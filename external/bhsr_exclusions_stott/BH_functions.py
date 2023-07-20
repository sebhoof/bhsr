#!/usr/bin/python
import os
import pickle

################ Stellar Index ################

# 0 - A 0620-00
# 1 - 4U 1543-475
# 2 - Cygnus X-1
# 3 - GRO J1655-40
# 4 - GRS 1915+105
# 5 - GW150914 (Primary)
# 6 - GW150914  (Secondary)
# 7 - GW151226 (Primary)
# 8 - GW151226 (Secondary)
# 9 - GW170104 (Primary)
# 10 - GW170104 (Secondary)
# 11 - LMC X-1
# 12 - LMC X-3
# 13 - M33 X-7
# 14 - XTE J1550-564

################################################

##############Supermassive Index ###############

# 0 - Ark 120
# 1 - Fairall
# 2 - MCG-6-30-1
# 3 - Mrk 110
# 4 - Mrk79
# 5 - Mrk335
# 6 - NGC 4051
# 7 - NGC 7469
# 8 - NGC3783
# 9 - M87*

################################################

bh_names = ['A_0620_00','BH4U_1543_475','Cygnus_X_1','GRO_J1655_40','GRS_1915_105','GW150914_PRI','GW150914_SEC','GW151226_PRI','GW151226_SEC','GW170104_PRI','GW170104_SEC','LMC_X_1','LMC_X_3','M33_X_7','XTE_J1550_564']
smbh_names = ['Ark_120','Fairall','MCG_6_30_1','Mrk_110','Mrk79','Mrk335','NGC_4051','NGC_7469','NGC3783']
stellar_functions = {}
supermassive_functions = {}

script_dir = os.path.dirname(os.path.realpath(__file__))
for i in range(len(bh_names)):
	with open(script_dir+'/STELLAR/FUNC/{0}_FUNC.npy'.format(bh_names[i]), 'rb') as f:
		stellar_functions[i] = pickle.load(f)

for i in range(len(smbh_names)):
	with open(script_dir+'/SUPERMASSIVE/FUNC/{0}_FUNC.npy'.format(smbh_names[i]), 'rb') as f:
		supermassive_functions[i] = pickle.load(f)

## Example usage
# import numpy as np
# mrange = np.arange(-14, -10, 0.05)
# pex = stellar_functions[13](mrange, 18)
# np.savetxt("pex_matt_m33x7.txt", np.array([mrange,pex]).T, fmt='%.6e')
