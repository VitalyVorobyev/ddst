""" """

mdn   = 1.86484
mdp   = mdn + 0.004822  # 1.86965
mdstn = mdn + 0.142014
mdstp = mdn + 0.1454257
mpin  = 0.1349766
mpip  = 0.13957018
gamma_star_n = 65.5e-6
gamma_star_p = 85.5e-6

br_dstp_dppin = 0.307  # D*+ -> D+ pi0
br_dstp_dnpip = 0.677  # D*+ -> D0 pi+
br_dstn_dnpin = 0.647  # D*0 -> D0 pi0
br_dstn_dngam = 0.353  # D*0 -> D0 gamma

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

# gs = (20 + 0.5j) * 10**-3
# gt = (20 + 0.5j) * 10**-3

gs = (30 + 0.5j) * 10**-3
gt = (-10 + 0.5j) * 10**-3

DalitzNBins = 1000
GammaScale = 0.032 / 7

#######################
##  D0D+pi0  options ##
#######################
# Turn on interference between D*0D+ and D*+D0 in the D0D+pi0 channel
interf_dndstp_dpdstn = True
# if False the [D*0 -> D0 pi0]D+ amplitude is excluded
include_dstndp = True
# if False the [D*+ -> D+ pi0]D0 amplitude is excluded
include_dstpdn = True
# 
include_dd_pwave = False
alpha_pwave = 0.
norm_pwave = 0.6*10**8
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
