import os, shutil, subprocess
import numpy as np
import matplotlib.pyplot as plt
import warnings

from dust_extinction import calzetti
from add_emlines import add_emission_lines
from igm_attenuation import inoue_tau

class TemplateSED_BC03(object):

    def __init__(self,
                 age, sfh, metallicity=None, input_ssp=None,
                 tau=None, Av=None, emlines=False, dust='none',
                 redshift=None, igm=False,
                 sfr=1, gasrecycle=False, epsilon=0.001, tcutsfr=20,
                 units='flambda', W1=1, W2=1e7,
                 imf='chab', res='hr', uid=None,
                 rootdir='galaxev/', library_version=2003, library='stelib',
                 workdir=None, cleanup=True, verbose=True):

        """
        metallicity:     0.0001(m22), 0.0004(m32), 0.004(m42), 0.004(m52), 0.02(m62), 0.05(m72) [BC2003 option]
        age:             0 < age < 13.5 Gyr [BC2003 option]
        sfh:             Star formation history [BC2003 option]
                            - 'constant':   constant SFR (requires SFR, TCUTSFR)
                            - 'exp':        exponentially declining  (requires TAU, TCUTSFR, GASRECYCLE[, EPSILON])
                            - 'ssp':        single stellar pop
                            - 'single':     single burst  (requires TAU - length of burst)
        tau:             e-folding timescale for exponentially declining SFH [BC2003 option]
        Av:              dust content (A_v)
        emlines:         adds emission lines
        dust:            dust extinction law
                            - 'none':       No dust extinction to be applied
                            - 'calzetti':   Apply Calzetti (2000) dust law
                            - 'cardelli':   Apply Cardelli (1989) dust law
        z:               Redshift for the SED
        igm:             Apply IGM attentuation?
        SFR:             Star formation rate [BC2003 option]
        gasrecycle:      Recycle the gas [BC2003 option]
        epsilon:         Fraction of recycled gas [BC2003 option]
        tcutsfr:         Time at which SFR drops to 0
        units:           Units to return the sed in [BC2003 option]
                            - 'lnu':       ergs/s/Hz
                            - 'llambda':   ergs/s/Angs
        W1, W2:          Limits of the wavelength range to compute the SED in [BC2003 option]
        imf:             Initial Mass Function ('salp' or 'chab') [BC2003 option]
        res:             Resolution of the SED [BC2003 option]
                            - 'hr':         High resolution
                            - 'lr':         Low resolution
        uid:             Unique ID for the SED
        rootdir:         Root directory for the GALAXEv installation
        workdir:         Working directory to store temporary files
        library_version: Specify which version of BC03 -- 2003, 2012
        library:         Specify specific library (only valid for 2012 version) -- 'stelib','BaSeL'
        input_ssp:       Option to directly specify what input ISED file to use
        cleanup:         Cleanup the temporary files?
        verbose:         Print messages to terminal?
        """

        self.sfh_key = {'ssp':0,'exp':1,'single':2,'constant':3}
        self.imf_dir_key = {'salp':'salpeter','chab':'chabrier','kroup':'kroupa'}
        self.metallicity_key = {0.0001:'m22',0.0004:'m32',0.004:'m42',0.008:'m52',0.02:'m62',0.05:'m72',0.1:'m82'}
        self.inv_metallicity_key = dict([[v,k] for k,v in self.metallicity_key.items()])

        self.res         = res
        self.age         = age
        self.sfh         = sfh
        self.sfr         = sfr
        self.tau         = tau
        self.gasrecycle  = gasrecycle
        self.epsilon     = epsilon
        self.tcutsfr     = tcutsfr
        self.Av          = Av
        self.emlines     = emlines
        self.dust        = dust
        self.redshift    = redshift
        self.igm         = igm
        self.units       = units
        self.W1          = W1
        self.W2          = W2
        self.rootdir     = rootdir
        self.library     = library
        self.library_version = library_version

        if input_ssp:
            warnings.warn('Ignoring IMF and Metallicity args and using provided input ISED file: %s' % input_ssp)
            self.input_ssp = input_ssp
            self.imf = input_ssp.split('_')[-2]
            self.metallicity = self.inv_metallicity_key[input_ssp.split('_')[-3]]
        else:
            self.metallicity = metallicity
            self.imf = imf

        self.Q = {}
        self.M_unnorm = {}

        self.read_age_input()
        self.check_input()

        if   not input_ssp and self.library_version==2003:
            self.input_ssp = 'bc2003_'+self.res+'_'+self.metallicity_key[self.metallicity]+'_'+self.imf+'_ssp'
        elif not input_ssp and self.library_version==2012:
            self.input_ssp = 'bc2003_'+self.res+'_'+self.library+'_'+self.metallicity_key[self.metallicity]+'_'+self.imf+'_ssp'

        self.model_dir = self.rootdir+'models/Padova1994/'+self.imf_dir_key[self.imf]+'/'
        self.workdir = workdir+'/' if workdir else os.getcwd()+'/'
        self.uid = uid
        self.ssp_output = self.uid+'_ssp'
        self.csp_output = self.uid+'_csp'
        self.cleanup = cleanup
        self.verbose = verbose

        self.define_env()
        self.mk_csp_input()
        self.mk_gpl_input()

    def read_age_input(self):

        if   self.library_version==2003: self.age_limit = 24
        elif self.library_version==2012: self.age_limit = 100

        if not (isinstance(self.age,np.ndarray) or isinstance(self.age,list)):
            self.ages = str(self.age)
            self.age = np.array([self.age,])
        elif len(self.age) <= self.age_limit:
            self.age = np.asarray(self.age)
            self.ages = ','.join(np.round(self.age,6).astype(str))
        else:
            raise Exception('Cannot provide more than %i ages!' % self.age_limit)

    def check_input(self):

        if any(self.age > 13.5):
            raise Exception("SED age (%s) provided is older than the Universe (13.5 Gyr)!" % str(self.age))
        if self.sfh not in self.sfh_key.keys():
            raise Exception("Incorrect SFH provided: "+str(self.sfh)+"\n" \
                            "Please choose from:"+str(self.sfh_key.keys()))
        if self.metallicity not in self.metallicity_key.keys():
            raise Exception("Incorrect metallicity provided: "+str(self.metallicity)+"\n" \
                            "Please choose from:"+str(self.metallicity_key.keys()))
        if self.imf not in self.imf_dir_key.keys():
            raise Exception("Incorrect IMF provided: "+str(self.imf)+"\n" \
                            "Please choose from:"+str(self.imf_dir_key.keys()))
        if self.res not in ['hr','lr']:
            raise Exception("Incorrect resolution provided: "+str(self.res)+"\n" \
                            "Please choose from: 'hr','lr'")
        if self.units not in ['flambda','fnu']:
            raise Exception("Incorrect flux units provided: "+str(self.units)+"\n" \
                            "Please choose from: 'flambda','fnu'")
        if self.dust not in ['none','calzetti','cardelli']:
            raise Exception("Incorrect dust law provided: "+str(self.dust)+"\n" \
                            "Please choose from: 'none','calzetti','cardelli'")
        if self.redshift is not None and self.redshift < 0:
            raise Exception("Incorrect redshift provided: "+str(self.redshift)+"\n" \
                            "Please provide a positive value.")
        if self.redshift is None and self.igm:
            warnings.warn("No redshift provided, and thus IGM attentuation cannot be applied.")
        if self.library_version not in [2003,2012]:
            raise Exception("Invalid library_version: "+str(self.library_version)+"\n" \
                            "Please choose from: 2003,2012")
        if self.library not in ['stelib','BaSeL']:
            raise Exception("Incorrect library choice: "+str(self.library)+"\n" \
                            "Please choose from: 'stelib','BaSeL'")

    def define_env(self):

        self.env_string = "export FILTERS="+self.rootdir+"src/FILTERBIN.RES;" \
                          "export A0VSED="+self.rootdir+"src/A0V_KURUCZ_92.SED;" \
                          "export RF_COLORS_ARRAYS="+self.rootdir+"src/RF_COLORS.filters;"
        if self.library_version == 2012:
            self.env_string += "export SUNSED="+self.rootdir+"src/SUN_KURUCZ_92.SED;"

    def generate_sed(self):

        self.do_bin_ised()

        self.do_csp()
        if self.cleanup: self.csp_cleanup()

        self.do_gpl()
        if self.cleanup: self.gpl_cleanup()

        self.read_gpl()
        if self.cleanup: self.post_gpl_cleanup()

        if self.emlines: self.add_emlines()
        if self.dust:    self.add_dust()

        if self.redshift: self.redshift_evo()

    def do_bin_ised(self):

        if   os.path.isfile(self.model_dir+self.input_ssp+'.ised'):
            shutil.copyfile(self.model_dir+self.input_ssp+'.ised',
                            self.workdir+self.ssp_output+'.ised')
        elif os.path.isfile(self.model_dir+self.input_ssp+'.ised_ASCII'):
            shutil.copyfile(self.model_dir+self.input_ssp+'.ised_ASCII',
                            self.workdir+self.ssp_output+'.ised_ASCII')
            if self.verbose:
                subprocess.call(self.rootdir+'src/bin_ised '+self.ssp_output+'.ised_ASCII',
                                cwd=self.workdir, shell=True)
            else:
                subprocess.call(self.rootdir+'src/bin_ised '+self.ssp_output+'.ised_ASCII',
                                cwd=self.workdir, shell=True, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
            os.remove(self.workdir+self.ssp_output+'.ised_ASCII')
        else:
            raise Exception('Template %s not found in %s.' % (self.input_ssp,self.model_dir))

    def mk_csp_input(self):

        self.csp_input = {}
        self.csp_input['CSPINPUT'] = self.ssp_output
        self.csp_input['DUST'] = 'N'
        self.csp_input['REDSHIFT'] = str(0)
        self.csp_input['SFHCODE'] = str(self.sfh_key[self.sfh])
        self.csp_input['SFR'] = str(self.sfr)
        self.csp_input['TAU'] = str(self.tau)
        self.csp_input['GASRECYCLE'] = 'Y' if self.gasrecycle else 'N'
        self.csp_input['EPSILON'] = str(self.epsilon)
        self.csp_input['TCUTSFR'] = str(self.tcutsfr)
        self.csp_input['CSPOUTPUT'] = self.csp_output

    def do_csp(self):

        csp_input_string =  self.csp_input['CSPINPUT'] + '\n'
        csp_input_string += self.csp_input['DUST'] +  '\n'
        if self.library_version==2012:
            csp_input_string += self.csp_input['REDSHIFT'] +  '\n'
        csp_input_string += self.csp_input['SFHCODE'] + '\n'

        if   self.sfh == 'ssp':
            pass
        elif self.sfh == 'exp':
            csp_input_string += self.csp_input['TAU'] + '\n'
            csp_input_string += self.csp_input['GASRECYCLE'] + '\n'
            if self.csp_input['GASRECYCLE'] == 'Y':
                csp_input_string += self.csp_input['EPSILON'] + '\n'
            csp_input_string += self.csp_input['TCUTSFR'] + '\n'
        elif self.sfh == 'single':
            csp_input_string += self.csp_input['TAU'] + '\n'
        elif self.sfh == 'constant':
            csp_input_string += self.csp_input['SFR'] + '\n'
            csp_input_string += self.csp_input['TCUTSFR'] + '\n'

        csp_input_string += self.csp_input['CSPOUTPUT'] + '\n'

        with open(self.workdir+self.uid+'_csp.in','w') as f: f.write(csp_input_string)
        if self.verbose:
            subprocess.call(self.env_string+self.rootdir+'src/csp_galaxev < '+self.uid+'_csp.in',
                            cwd=self.workdir, shell=True)
        else:
            subprocess.call(self.env_string+self.rootdir+'src/csp_galaxev < '+self.uid+'_csp.in',
                            cwd=self.workdir, shell=True, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
        os.remove(self.workdir+self.uid+'_csp.in')

    def csp_cleanup(self):

        os.remove(self.workdir+self.csp_output+'.1ABmag')
        os.remove(self.workdir+self.csp_output+'.1color')
        os.remove(self.workdir+self.csp_output+'.2color')
        os.remove(self.workdir+self.csp_output+'.5color')
        os.remove(self.workdir+self.csp_output+'.6lsindx_ffn')
        os.remove(self.workdir+self.csp_output+'.6lsindx_sed')
        os.remove(self.workdir+self.csp_output+'.6lsindx_sed_lick_system')
        os.remove(self.workdir+self.csp_output+'.7lsindx_ffn')
        os.remove(self.workdir+self.csp_output+'.7lsindx_sed')
        os.remove(self.workdir+self.csp_output+'.7lsindx_sed_lick_system')
        os.remove(self.workdir+self.csp_output+'.8lsindx_sed_fluxes')
        os.remove(self.workdir+'bc03.rm')

        if self.library_version==2012:
            os.remove(self.workdir+self.csp_output+'.9color')
            os.remove(self.workdir+self.csp_output+'.acs_wfc_color')
            os.remove(self.workdir+self.csp_output+'.legus_uvis1_color')
            os.remove(self.workdir+self.csp_output+'.wfc3_color')
            os.remove(self.workdir+self.csp_output+'.wfc3_uvis1_color')
            os.remove(self.workdir+self.csp_output+'.wfpc2_johnson_color')
            os.remove(self.workdir+self.csp_output+'.w_age_rf')
            os.remove(self.workdir+'fort.24')

    def mk_gpl_input(self):

        self.gpl_input = {}
        self.gpl_input['GPLINPUT'] = self.csp_output
        if   self.units == 'flambda':  self.gpl_input['W1W2W0F0Z'] =  str(self.W1)+','+str(self.W2)
        elif self.units == 'fnu':      self.gpl_input['W1W2W0F0Z'] = str(-self.W1)+','+str(self.W2)
        self.gpl_input['AGES'] = self.ages
        self.gpl_input['GPLOUTPUT'] = self.csp_output+'.spec'

    def do_gpl(self):

        gpl_input_string =  self.gpl_input['GPLINPUT'] + '\n'

        if self.library_version==2003:
            gpl_input_string += self.gpl_input['W1W2W0F0Z'] + '\n'
            gpl_input_string += self.gpl_input['AGES'] + '\n'
        elif self.library_version==2012:
            gpl_input_string += self.gpl_input['AGES'] + '\n'
            gpl_input_string += self.gpl_input['W1W2W0F0Z'] + '\n'

        gpl_input_string += self.gpl_input['GPLOUTPUT'] + '\n'

        with open(self.workdir+self.uid+'_gpl.in','w') as f: f.write(gpl_input_string)
        if self.verbose:
            subprocess.call(self.rootdir+'src/galaxevpl < '+self.uid+'_gpl.in',
                            cwd=self.workdir, shell=True)
        else:
            subprocess.call(self.rootdir+'src/galaxevpl < '+self.uid+'_gpl.in',
                            cwd=self.workdir, shell=True, stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
        os.remove(self.workdir+self.uid+'_gpl.in')

    def gpl_cleanup(self):

        os.remove(self.workdir+self.ssp_output+'.ised')
        os.remove(self.workdir+self.csp_output+'.ised')

    def read_gpl(self):

        dtype = [('waves',float),]+[('spec%i'%(i+1),float) for i in range(len(self.age))]
        self.sed = np.genfromtxt(self.workdir+self.csp_output+'.spec',dtype=dtype)
        age3, Q = np.genfromtxt(self.workdir+self.csp_output+'.3color', usecols=(0,5), unpack=True)
        age4, M = np.genfromtxt(self.workdir+self.csp_output+'.4color', usecols=(0,6), unpack=True)

        for x,age in zip(self.sed.dtype.names[1:],self.age):

            self.sed[x] = self.sed[x] * 3.839e33

            log_age = np.log10(age*1e9)
            diff = abs(age3 - log_age)
            self.Q[x] = Q[diff == min(diff)][0]

            diff = abs(age4 - log_age)
            self.M_unnorm[x] = M[diff == min(diff)][0]

    def post_gpl_cleanup(self):

        os.unlink(self.workdir+self.csp_output+'.spec')
        os.unlink(self.workdir+self.csp_output+'.3color')
        os.unlink(self.workdir+self.csp_output+'.4color')

    def add_emlines(self):

        for x in self.sed.dtype.names[1:]:
            self.sed[x] = add_emission_lines(self.sed['waves'], self.sed[x], self.Q[x], self.metallicity, self.units)

    def add_dust(self):

        for x in self.sed.dtype.names[1:]:
            if   self.dust == 'calzetti':
                self.sed[x] = self.sed[x] * np.exp(-calzetti(self.sed['waves'], self.Av))
            elif self.dust == 'cardelli':
                self.sed[x] = self.sed[x] * np.exp(-cardelli(self.sed['waves'], self.Av))

    def redshift_evo(self):

        self.sed['waves'] *= (1+self.redshift)
        if self.igm:
            for x in self.sed.dtype.names[1:]:
                self.sed[x] = self.sed[x] * np.exp(-inoue_tau(self.sed['waves'], self.redshift))

    def plot_sed(self,save=None):

        fig, ax = plt.subplots(1,1,figsize=(15,8),dpi=75,tight_layout=True)

        colors = plt.cm.gist_rainbow_r(np.linspace(0.1,0.95,len(self.sed.dtype.names[1:])))
        for age,x,c in zip(self.age,self.sed.dtype.names[1:],colors):
            ax.plot(self.sed['waves'],self.sed[x],c=c,lw=1.5,alpha=0.8,label="Age = %.3f Gyr" % age)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Rest-frame Wavelength ($\AA$)')
        if self.units == 'flambda': ax.set_ylabel(r'F$_\lambda$ [ergs/s/$\AA$]')
        elif self.units == 'fnu':   ax.set_ylabel(r'F$_\nu$ [ergs/s/Hz]')

        title = "BC2003 SED -- " \
                "IMF="+self.imf_dir_key[self.imf]+", " \
                "Z="+str(self.metallicity)+", " \
                "SFH="+self.sfh+", "
        if self.sfh == 'exp': title += r"$\tau$="+str(self.tau)+", "
        if self.sfh == 'single': title += r"$\Delta$="+str(self.tau)+", "
        title += "Av="+str(self.Av)+", " if self.dust else "Av=0, "
        title += "EmLines="+str(self.emlines)

        ax.set_title(title)
        ax.legend(fontsize=16)
        if save: fig.savefig(save)
        else: plt.show()
