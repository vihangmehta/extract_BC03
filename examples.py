import numpy as np
import matplotlib.pyplot as plt
from extract_bc03 import TemplateSED_BC03

def plot():

    template = TemplateSED_BC03(metallicity=0.02, age=[1,2,3,5,10], sfh='exp', tau=2, Av=0,
                                #input_ised='bc2003_lr_BaSeL_m52_chab_ssp.ised',
                                dust='calzetti', emlines=False,
                                redshift=3, igm=True,
                                imf='chab', res='lr', uid='test', units='flambda',
                                rootdir='/data/highzgal/mehta/Software/galaxev12/', workdir='.',
                                library_version=2012,
                                cleanup=True,verbose=False)
    template.generate_sed()

    fig, ax = plt.subplots(1,1,figsize=(15,8),dpi=75,tight_layout=True)

    cind = np.linspace(0.2,0.95,5)
    for x,c in zip(template.sed.dtype.names[1:],plt.cm.Greys(cind)):
        ax.plot(template.sed['waves'],template.sed[x],c=c,lw=1.25,alpha=0.8)

    template.emlines = True
    template.generate_sed()
    for x,c in zip(template.sed.dtype.names[1:],plt.cm.Blues(cind)):
        ax.plot(template.sed['waves'],template.sed[x],c=c,lw=1.25,alpha=0.8)

    template.Av = 1
    template.generate_sed()
    for x,c in zip(template.sed.dtype.names[1:],plt.cm.Reds(cind)):
        ax.plot(template.sed['waves'],template.sed[x],c=c,lw=1.25,alpha=0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rest-frame Wavelength [$\AA$]')
    ax.set_ylabel('L$_\lambda$ [ergs/s/$\AA$]')
    ax.set_xlim(1e2,1e8)
    ax.set_ylim(1e17,1e31)
    plt.show()

if __name__ == '__main__':

    plot()
