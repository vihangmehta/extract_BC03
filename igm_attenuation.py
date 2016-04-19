import numpy as np

inoue_coeffs = np.genfromtxt('inoue_coeffs.dat',dtype=[('j',int),('lambda',float),('A1_LAF',float),('A2_LAF',float),('A3_LAF',float),('A1_DLA',float),('A2_DLA',float)],unpack=True)

#INOUE IGM ATTENUATION
def inoue_tau(l,z):

    if not isinstance(l, np.ndarray):
        if isinstance(l, list): l = np.array(l)
        else: l = np.array([l])

    ly_l = 912.
    obs_l = l*(1.+z)
    A, B = 1+z, obs_l/ly_l
    tau_LAF_LS, tau_DLA_LS, tau_LAF_LC, tau_DLA_LC = np.zeros(len(l)), np.zeros(len(l)), np.zeros(len(l)), np.zeros(len(l))

    # Add up contributions from each line in LyS
    for x in inoue_coeffs:

        # LAF Component for LyS
        cond1 = (obs_l > x['lambda']) & (obs_l < x['lambda']*(1+z)) & (obs_l < 2.2*x['lambda'])
        cond2 = (obs_l > x['lambda']) & (obs_l < x['lambda']*(1+z)) & (obs_l >= 2.2*x['lambda']) & (obs_l < 5.7*x['lambda'])
        cond3 = (obs_l > x['lambda']) & (obs_l < x['lambda']*(1+z)) & (obs_l >= 5.7*x['lambda'])
        tau_LAF_LS[cond1] += x['A1_LAF']*(obs_l[cond1]/x['lambda'])**1.2
        tau_LAF_LS[cond2] += x['A2_LAF']*(obs_l[cond2]/x['lambda'])**3.7
        tau_LAF_LS[cond3] += x['A3_LAF']*(obs_l[cond3]/x['lambda'])**5.5

        # DLA Component for LyS
        cond1 = (obs_l > x['lambda']) & (obs_l < x['lambda']*(1+z)) & (obs_l < 3.0*x['lambda'])
        cond2 = (obs_l > x['lambda']) & (obs_l < x['lambda']*(1+z)) & (obs_l >= 3.0*x['lambda'])
        tau_DLA_LS[cond1] += x['A1_DLA']*(obs_l[cond1]/x['lambda'])**2.0
        tau_DLA_LS[cond2] += x['A2_DLA']*(obs_l[cond2]/x['lambda'])**3.0

    # LAF Component for LyC
    if z < 1.2:
        cond = (obs_l < A*ly_l)
        tau_LAF_LC[cond] += 0.325*(B[cond]**1.2 - (A**-0.9)*(B[cond]**2.1))
    elif z >= 1.2 and z < 4.7:
        cond1 = (obs_l < 2.2*ly_l)
        cond2 = (obs_l >= 2.2*ly_l) & (obs_l < A*ly_l)
        tau_LAF_LC[cond1] += (2.55e-2*(A**1.6)*(B[cond1]**2.1)) + (0.325*B[cond1]**1.2) - (0.250*B[cond1]**2.1)
        tau_LAF_LC[cond2] += 2.55e-2*((A**1.6)*(B[cond2]**2.1) - B[cond2]**3.7)
    elif z >= 4.7:
        cond1 = (obs_l < 2.2*ly_l)
        cond2 = (obs_l >= 2.2*ly_l) & (obs_l < 5.7*ly_l)
        cond3 = (obs_l >= 5.7*ly_l) & (obs_l < A*ly_l)
        tau_LAF_LC[cond1] += (5.22e-4*(A**3.4)*(B[cond1]**2.1)) + (0.325*B[cond1]**1.2) - (3.14e-2*B[cond1]**2.1)
        tau_LAF_LC[cond2] += (5.22e-4*(A**3.4)*(B[cond2]**2.1)) + (0.218*B[cond2]**2.1) - (2.55e-2*B[cond2]**3.7)
        tau_LAF_LC[cond3] += 5.22e-4*((A**3.4)*(B[cond3]**2.1) - B[cond3]**5.5)
    else: print 'Error in LAF component of tau for LyC'

    # DLA Component for LyC
    if z < 2.0:
        cond = (obs_l < A*ly_l)
        tau_DLA_LC[cond] += (0.211*A**2.0) - (7.66e-2*(A**2.3)*(B[cond]**-0.3)) - (0.135*B[cond]**2.0)
    elif z >= 2.0:
        cond1 = (obs_l < 3.0*ly_l)
        cond2 = (obs_l >=3.0*ly_l) & (obs_l < A*ly_l)
        tau_DLA_LC[cond1] = 0.634 + (4.7e-2*A**3.0) - (1.78e-2*(A**3.3)*(B[cond1]**-0.3)) - (0.135*B[cond1]**2.0) - (0.291*B[cond1]**-0.3)
        tau_DLA_LC[cond2] = (4.70e-2*A**3.0) - (1.78e-2*(A**3.3)*(B[cond2]**-0.3)) - (2.92e-2*B[cond2]**3.0)
    else: print 'Error in DLA component of tau for LyC'

    return tau_LAF_LS + tau_DLA_LS + tau_LAF_LC + tau_DLA_LC
