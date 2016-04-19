import numpy as np

# CARDELLI EXTINCTION LAW
def cardelli(l,Av):

    Rv = 3.1
    x = 1e4 / l
    a, b, k = np.zeros(len(l)), np.zeros(len(l)), np.zeros(len(l))

    cond1 = (0.3 <= x) & (x <= 1.1)
    a[cond1] = 0.574*x[cond1]**1.61
    b[cond1] = -0.527*x[cond1]**1.61

    y = (x - 1.82)
    cond2 = (1.1 <= x) & (x <= 3.3)
    a[cond2] = 1 + 0.17699*y[cond2] - 0.50447*(y[cond2]**2) - 0.02427*(y[cond2]**3) + 0.72085*(y[cond2]**4) + 0.01979*(y[cond2]**5) - 0.77530*(y[cond2]**6) + 0.32999*(y[cond2]**7)
    b[cond2] = 0 + 1.41338*y[cond2] + 2.28305*(y[cond2]**2) + 1.07233*(y[cond2]**3) - 5.38434*(y[cond2]**4) - 0.62251*(y[cond2]**5) + 5.30260*(y[cond2]**6) - 2.09002*(y[cond2]**7)

    cond3 = (3.3 <= x) & (x < 5.9)
    a[cond3] = 1.752 - 0.316*x[cond3] - 0.104 / ((x[cond3]-4.67)**2 + 0.341)
    b[cond3] = -3.090 + 1.825*x[cond3] + 1.206 / ((x[cond3]-4.62)**2 + 0.263)

    cond4 = (5.9 <= x) & (x <= 8)
    a[cond4] = 1.752 - 0.316*x[cond4] - 0.104 / ((x[cond4]-4.67)**2 + 0.341) - 0.04473*(x[cond4]-5.9)**2 - 0.009779*(x[cond4]-5.9)**3
    b[cond4] = -3.090 + 1.825*x[cond4] + 1.206 / ((x[cond4]-4.62)**2 + 0.263) + 0.213*(x[cond4]-5.9)**2 + 0.1207*(x[cond4]-5.9)**3

    cond5 = (8 <= x) & (x <= 10)
    a[cond5] = -1.073 - 0.628*(x[cond5]-8) + 0.137*(x[cond5]-8)**2 - 0.070*(x[cond5]-8)**3
    b[cond5] = 13.670 + 4.257*(x[cond5]-8) - 0.420*(x[cond5]-8)**2 + 0.374*(x[cond5]-8)**3

    ### ADDED for completeness ###
    cond6 = (10 < x)
    x[cond6] = 10. # <------
    a[cond6] = -1.073 - 0.628*(x[cond6]-8) + 0.137*(x[cond6]-8)**2 - 0.070*(x[cond6]-8)**3
    b[cond6] = 13.670 + 4.257*(x[cond6]-8) - 0.420*(x[cond6]-8)**2 + 0.374*(x[cond6]-8)**3

    k = a + (b / Rv)
    tau = np.log(10) * ( 0.4 * Av * k )
    return tau

# CALZETTI EXTINCTION LAW
def calzetti(l,Av):

    Rv = 4.05
    x = 1e4 / l
    k = np.zeros(len(l))

    cond1 = (0.63 <= l/1e4) & (l/1e4 <= 2.20)
    k[cond1] = 2.659*(-1.857 + 1.040*x[cond1]) + Rv

    cond2 = (0.12 <= l/1e4) & (l/1e4 < 0.63)
    k[cond2] = 2.659*(-2.156 + 1.509*x[cond2] - 0.198*x[cond2]**2 + 0.011*x[cond2]**3) + Rv

    ### ADDED for completeness ###
    cond3 = (l/1e4 < 0.12)
    x[cond3] = 1./0.12
    k[cond3] = 2.659*(-2.156 + 1.509*x[cond3] - 0.198*x[cond3]**2 + 0.011*x[cond3]**3) + Rv

    EBV = Av / Rv
    tau = np.log(10) * (0.4 * EBV * k)
    return tau