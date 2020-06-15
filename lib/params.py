""" """
import numpy as np

mdn    = 1.86484
dmdp   = 0.004822
dmdstn = 0.142014
dmdstp = 0.1454257
mpin   = 0.1349766
mpip   = 0.13957018
gamma_star_n = 58.3e-6
gamma_star_p = 85.5e-6
mdp   = mdn + dmdp  # 1.86965
mdstn = mdn + dmdstn
mdstp = mdn + dmdstp

isospin_breaking = 1.0
mdp   = mdn   + isospin_breaking * (mdp   - mdn)
mdstp = mdstn + isospin_breaking * (mdstp - mdstn)
mpip  = mpin  + isospin_breaking * (mpip  - mpin)
gamma_star_p = gamma_star_n + isospin_breaking * (gamma_star_p - gamma_star_n)

br_dstp_dppin = 0.307  # D*+ -> D+ pi0
br_dstp_dnpip = 0.677  # D*+ -> D0 pi+
br_dstn_dnpin = 0.647  # D*0 -> D0 pi0
br_dstn_dngam = 0.353  # D*0 -> D0 gamma

# Energy-dependent width of D*0
delta000 = mdstn - mdn - mpin
mdstnSq = mdstn**2
gamma_star_n_dnpin = gamma_star_n * br_dstn_dnpin
gamma_star_n_dngam = gamma_star_n * br_dstn_dngam

# Resolution parameters
sigma_mdn = 8.20 * 10**-3  # m(D0) LHCb resolution
sigma_ppi = 1.17 * 10**-3  # m(D*+) LHCb resolution

# gs = ( 30 + 1.j) * 10**-3
# gt = (-30 + 1.j) * 10**-3

# gs = (35 + 0.5j) * 10**-3
# gs = (50    + 0.5j) * 10**-3
# gt = (10000 + 0.5j) * 10**-3

# gs = (25 + 0.5j) * 10**-3
# gt = (50 + 0.5j) * 10**-3

# gs = (30 + 0.5j) * 10**-3
# gt = (-50 + 0.5j) * 10**-3

# gs = (25 + 0.5j) * 10**-3
# gt = (0 + 0.5j) * 10**-3

# gs = (60 + 2.3j) * 10**-3
# gt = (10000 + 2.3j) * 10**-3

# gs = (23 +3j) * 10**-3
# gt = (13 +3j) * 10**-3

gs = (   43 + 1.5j) * 10**-3
gt = (25000 + 1.5j) * 10**-3

DalitzNBins = 512
GammaScale = 0.024

#######################
##  D0D+pi0  options ##
#######################
interf_dndstp_dpdstn = True  # Turn on interference between D*0D+ and D*+D0 in the D0D+pi0 channel
include_dstndp = True  # if False the [D*0 -> D0 pi0]D+ amplitude is excluded
include_dstpdn = True  # if False the [D*+ -> D+ pi0]D0 amplitude is excluded
include_dd_pwave = True
alpha_pwave = 0.
norm_pwave = 0.8*10**11
pwave_over_dndnpip = 0.2
g1 = 1.
g2 = 1.
Rin = 1e-3
phiin = 1.*np.pi
#######################

datapath = '/home/vitaly/work/lhcb/DDpi'
datafile = '/'.join([datapath, 'all.root'])

if __name__ == '__main__':
    print('m(D0) + m(D*+): {:.3f}'.format(mdn + mdstp))
    print('m(D+) + m(D*0): {:.3f}'.format(mdp + mdstn))
    print('m(D*0): {:.3f}'.format(mdstn))
    print('m(D*+): {:.3f}'.format(mdstp))
    print('m^2(D*0): {:.3f}'.format(mdstn**2))
    print('m^2(D*+): {:.3f}'.format(mdstp**2))
