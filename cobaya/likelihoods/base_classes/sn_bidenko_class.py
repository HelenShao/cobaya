import numpy as np
import os
import pandas as pd
import numpy as np
import scipy.linalg as la
import astropy.constants
import scipy.integrate as intgr
import pandas as pn

# Local
from cobaya.log import LoggedError
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.likelihood import Likelihood

_twopi = 2 * np.pi

use_abs_mag = True
if use_abs_mag:
    post_record_file = '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/LCDM/cobaya_sn_like.txt'
else:
    post_record_file = '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/LCDM/cobaya_sn_lik_noMb.txt'
file_like = open(post_record_file, 'w')

class SN(Likelihood):

    def initialize(self):
        data_file = "/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat"
        b=pn.read_csv(data_file,delimiter = ' ')
        
        cov_file = '/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov'
        self.cov = np.reshape(pn.read_table(cov_file,sep=' ').values,(1701,1701))
        
        self.mb = b['m_b_corr'].values
        self.z = b['zHD'].values
        self.zhel = b['zHEL'].values
        
        # define here redshift limits (e.g. to use upper redshift cut)
        self.cccc = ((b['zHD']>0.01) & (b['zHD']<10.5) | (b['IS_CALIBRATOR']==1)).values == False
        
        self.cond1 = np.where(self.cccc==True)
        self.cond2 = np.where(self.cccc==False)
        
        # applying redshift limits to datapoints, covariance and distance matrices 
        self.cov = np.delete(self.cov,self.cond1[0],1)
        self.cov = np.delete(self.cov,self.cond1[0],0)
        self.z = self.z[self.cond2]
        self.mb = self.mb[self.cond2]
        self.zhel = self.zhel[self.cond2]
         
        # defining positions of calibration SNe and distances to them
        self.sel_cal = b['IS_CALIBRATOR'].values[self.cond2]==1
        self.ceph_dist = b['CEPH_DIST'].values[self.cond2]
        self.det =  np.linalg.slogdet(self.cov.copy())[1]
        self.cov = la.cholesky(self.cov.copy(), lower=True, overwrite_a=True)


    def get_requirements(self):
        # State requisites to the theory code
        reqs = {"angular_diameter_distance": {"z": self.z}}
        
        ### True for cosmosis
        if self.use_abs_mag:
            reqs["Mb"] = None

        return reqs
        #return {}

    def logp(self, **params_values):
        
        # getting current parameter from sampler
        #H0gp = params_values['H0']
        #omegamgp = params_values['Omega_m']
        Mgp = params_values['Mb']

        '''
        # calculating cosmological distances
        moduli = []
        for  i in range(len(self.z)):
            moduli.append(da(self.z[i], H0gp,omegamgp) * (1. + self.z[i])*(1. + self.zhel[i]))
        moduli = np.array(moduli)
        moduli = 5 * np.log10(moduli) + 25
        '''

        ### calculate theory
        #angular_diameter_distances = \
            #self.provider.get_angular_diameter_distance(self.z)
        #moduli = (5 * np.log10((1 + self.zhel) * (1 + self.z) *
                                 #angular_diameter_distances)) + 25
        #moduli = np.array(moduli)

        moduli = []
        for  i in range(len(self.z)):
            moduli.append(angular_diameter_distances[i] * (1. + self.z[i])*(1. + self.zhel[i]))
        moduli = np.array(moduli)
        moduli = 5 * np.log10(moduli) + 25

        # replacing distances to calibration SNe with distance measurements
        moduli[self.sel_cal] = self.ceph_dist[self.sel_cal]
        
        mb = self.mb
        residuals = mb - Mgp
        residuals -= moduli

        residuals = la.solve_triangular(self.cov.copy(), residuals, lower=True, check_finite=False)
     
        chi2 = (residuals**2).sum()
        return -0.5 * chi2 - 1./2. * self.det