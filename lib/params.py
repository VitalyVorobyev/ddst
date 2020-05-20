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
br_dstn_dnnin = 0.647  # D*0 -> D0 pi0

if __name__ == '__main__':
    print('m(D0) + m(D*+): {:.3f}'.format(mdn + mdstp))
    print('m(D+) + m(D*0): {:.3f}'.format(mdp + mdstn))
    print('m(D*0): {:.3f}'.format(mdstn))
    print('m(D*+): {:.3f}'.format(mdstp))
    print('m^2(D*0): {:.3f}'.format(mdstn**2))
    print('m^2(D*+): {:.3f}'.format(mdstp**2))
