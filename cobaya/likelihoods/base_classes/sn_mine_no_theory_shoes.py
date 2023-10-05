# Pantheon+ as implemented in cosmosis
# Assuming m_b_corr has alpha/beta corr 
# Only doing Mb correction (to theory) if sampled

import numpy as np
import os
import pandas as pd
import scipy.linalg as la
import astropy.constants
import scipy.integrate as intgr

# Local
from cobaya.log import LoggedError
from cobaya.likelihoods.base_classes import DataSetLikelihood

_twopi = 2 * np.pi

use_abs_mag = True
record_like = False  #For evaluating likelihoods
if record_like:
    if use_abs_mag:
        post_record_file = '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/LCDM/cobaya_sn_like.txt'
    else:
        post_record_file = '/scratch/gpfs/hshao/CMB_Project/ACT/reproducing_pantheonP/LCDM/cobaya_sn_lik_noMb.txt'
    file_like = open(post_record_file, 'w')

# define functions for distance calculations
def h_z(z,*args):
    H0,om,pwr = args
    ol  = 1 - om
    h = ( H0 * ( om * (1+z)**3 + ol ) ** 0.5 ) ** pwr
    return h 
def da(z, H0,om,pwr = -1):
    d = intgr.quad(h_z,0.,z,args=(H0,om,pwr))[0]*(astropy.constants.c.value/1000.)/(1+z)
    return d

class SN(DataSetLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "SN"

    install_options = {"github_repository": "CobayaSampler/sn_data",
                       "github_release": "v1.3"}

    def init_params(self, ini):

        ### twoscriptmfit = F
        self.twoscriptmfit = ini.bool('twoscriptmfit')
        if self.twoscriptmfit:
            scriptmcut = ini.float('scriptmcut', 10.)

        ### intrinsicdisp = 0
        assert not ini.float('intrinsicdisp', 0) and not ini.float('intrinsicdisp0', 0)

        ### In jla.yaml only
        if getattr(self, "alpha_beta_names", None) is not None:
            self.alpha_name = self.alpha_beta_names[0]
            self.beta_name = self.alpha_beta_names[1]
    
        ### Pecz = 0, = 0.001 if not found in .dataset file
        self.pecz = ini.float('pecz', 0.001)
        #print("DEBUG: pecz = %.3f"%self.pecz)

        cols = None
        self.has_third_var = False

        ### full_long.dataset: data_file = Pantheon/lcparam_full_long_zhel.txt
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self.log.debug('Reading %s' % data_file)

        supernovae = {}
        self.names = []
        ix = 0
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' in line:
                    cols = line[1:].split()
                    ### Rename variable names from dataset file
                    for rename, new in zip(
                             ["c", "x1", 'm_b_corr_err_DIAG', 'zHD',  "zHEL", 
                             'zHDERR', 'x1ERR', 'cERR'], 
                            ['colour', 'stretch', 'dmb', 'zcmb', 'zhel',
                             'dz', 'dx1', 'dcolor']):
                        if rename in cols:
                            cols[cols.index(rename)] = new

                    ### self.has_third_var, has_x0_cov = False
                    self.has_third_var = 'third_var' in cols
                    has_x0_cov = 'cov_s_x0' in cols

                    zeros = np.zeros(len(lines) - 1)
                    self.third_var = zeros.copy()
                    self.dthird_var = zeros.copy()
                    self.set = zeros.copy()
                    for col in cols:
                        setattr(self, col, zeros.copy())

                # Read subsequent lines after header, counting number of SN
                elif line.strip():
                    if cols is None:
                        raise LoggedError(self.log, 'Data file must have comment header')
                    vals = line.split()
                    for i, (col, val) in enumerate(zip(cols, vals)):
                        if col == 'CID':
                            supernovae[val] = ix
                            self.names.append(val)
                        elif col != 'IDSURVEY':
                            getattr(self, col)[ix] = np.float64(val)
                    ix += 1
        ### False
        if has_x0_cov:
            sf = - 2.5 / (self.x0 * np.log(10))
            self.cov_mag_stretch = self.cov_s_x0 * sf
            self.cov_mag_colour = self.cov_c_x0 * sf
        
        ### Cosmosis z selected data
        #data_file = "/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat"
        #data = pd.read_csv(data_file,delim_whitespace=True)
        #self.origlen = len(data)

        #self.ww = ((data['zHD']>0.01) & (data['zHD']<10.5) | (data['IS_CALIBRATOR']==1))
        #self.zCMB = self.zcmb[self.ww] #use the vpec corrected redshift for zCMB 
        #self.zHEL = self.zhel[self.ww]
        #self.mag = self.m_b_corr[self.ww]
        #self.is_calibrator = data['IS_CALIBRATOR'][self.ww]
        #self.cepheid_distance = data['CEPH_DIST'][self.ww]

        data_file = "/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat"
        b=pd.read_csv(data_file,delimiter = ' ')
        self.origlen = len(b)
        
        #cov_file = '/home/hshao/codes_and_likes_v5/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov'
        #self.cov = np.reshape(pn.read_table(cov_file,sep=' ').values,(1701,1701))
        
        self.mb = b['m_b_corr'].values
        self.z = b['zHD'].values
        self.zhel = b['zHEL'].values
        
        # define here redshift limits (e.g. to use upper redshift cut)
        self.cccc = ((b['zHD']>0.01) & (b['zHD']<10.5) | (b['IS_CALIBRATOR']==1)).values == False
        self.ww = ((b['zHD']>0.01) & (b['zHD']<10.5) | (b['IS_CALIBRATOR']==1))

        self.cond1 = np.where(self.cccc==True)
        self.cond2 = np.where(self.cccc==False)
        
        # applying redshift limits to datapoints, covariance and distance matrices 
        #self.cov = np.delete(self.cov,self.cond1[0],1)
        #self.cov = np.delete(self.cov,self.cond1[0],0)
        self.zCMB = self.z[self.cond2]
        self.mag = self.mb[self.cond2]
        self.zHEL = self.zhel[self.cond2]

        # defining positions of calibration SNe and distances to them
        self.is_calibrator = b['IS_CALIBRATOR'].values[self.cond2]==1
        self.cepheid_distance = b['CEPH_DIST'].values[self.cond2]
        #self.det =  np.linalg.slogdet(self.cov.copy())[1]
        #self.cov = la.cholesky(self.cov.copy(), lower=True, overwrite_a=True)

        ### Uncertainties squared
        ### dz, dmb found in lcparam_full_long_zhel.txt
        ### dx1, dcolor, dthird_var = 0 in lcparam_full_long_zhel.txt
        #self.z_var = self.dz[self.ww] ** 2
        #self.mag_var = self.dmb[self.ww] ** 2
        #self.stretch_var = self.dx1[self.ww] ** 2
        #self.colour_var = self.dcolor[self.ww] ** 2
        #self.thirdvar_var = self.dthird_var[self.ww] ** 2

        ### Check number of SN
        self.nsn = ix
        self.log.debug('Number of SN read: %s ' % self.nsn)

        ### self.twoscriptmfit=False 
        if self.twoscriptmfit and not self.has_third_var:
            raise LoggedError(
                self.log, 'twoscriptmfit was set but thirdvar information not present')
        ### absdist_file=false
        if ini.bool('absdist_file'):
            raise LoggedError(self.log, 'absdist_file not supported')
        
        covmats = [
            'mag', 'stretch', 'colour', 'mag_stretch', 'mag_colour', 'stretch_colour']
        self.covs = {}

        ### Pantheon: has_mag_covmat = T --> read mag_covmat_file in full_long.dataset
        for name in covmats:
            if ini.bool('has_%s_covmat' % name):
                self.log.debug('Reading covmat for: %s ' % name)
                #print('Reading covmat for: %s ' % name)
                self.covs[name] = self._read_covmat(
                    os.path.join(self.path, ini.string('%s_covmat_file' % name)))
        
        ### self.alphabeta_covmat=False
        ### Since only one covmat item ('mag')
        self.alphabeta_covmat = (len(self.covs.items()) > 1 or
                                 self.covs.get('mag', None) is None)

        self._last_alpha = np.inf
        self._last_beta = np.inf

        ### In jla_lite.yaml only
        self.marginalize = getattr(self, "marginalize", False)

        assert self.covs

        ### Sys uncertainties, for diag of covmat - NOT USED FOR COSMOSIS
        # jla_prep
        zfacsq = 25.0 / np.log(10.0) ** 2
        #self.pre_vars = self.mag_var + zfacsq * self.pecz ** 2 * (
                #(1.0 + self.zCMB) / (self.zCMB * (1 + 0.5 * self.zCMB))) ** 2
        
        ### False
        if self.twoscriptmfit:
            A1 = np.zeros(self.nsn)
            A2 = np.zeros(self.nsn)
            # noinspection PyUnboundLocalVariable
            A1[self.third_var <= scriptmcut] = 1
            A2[self.third_var > scriptmcut] = 1
            has_A1 = np.any(A1)
            has_A2 = np.any(A2)
            if not has_A1:
                # swap
                A1 = A2
                A2 = np.zeros(self.nsn)
                has_A2 = False
            if not has_A2:
                self.twoscriptmfit = False
            self.A1 = A1
            self.A2 = A2

        ### False
        if self.marginalize:
            self.step_width_alpha = self.marginalize_params['step_width_alpha']
            self.step_width_beta = self.marginalize_params['step_width_beta']
            _marge_steps = self.marginalize_params['marge_steps']
            self.alpha_grid = np.empty((2 * _marge_steps + 1) ** 2)
            self.beta_grid = self.alpha_grid.copy()
            _int_points = 0
            for alpha_i in range(-_marge_steps, _marge_steps + 1):
                for beta_i in range(-_marge_steps, _marge_steps + 1):
                    if alpha_i ** 2 + beta_i ** 2 <= _marge_steps ** 2:
                        self.alpha_grid[_int_points] = (
                                self.marginalize_params['alpha_centre'] +
                                alpha_i * self.step_width_alpha)
                        self.beta_grid[_int_points] = (
                                self.marginalize_params['beta_centre'] +
                                beta_i * self.step_width_beta)
                        _int_points += 1
            self.log.debug('Marignalizing alpha, beta over %s points' % _int_points)
            self.marge_grid = np.empty(_int_points)
            self.int_points = _int_points
            self.alpha_grid = self.alpha_grid[:_int_points]
            self.beta_grid = self.beta_grid[:_int_points]
            self.invcovs = np.empty(_int_points, dtype=object)
            if self.precompute_covmats:
                for i, (alpha, beta) in enumerate(zip(self.alpha_grid, self.beta_grid)):
                    self.invcovs[i] = self.inverse_covariance_matrix(alpha, beta)
        
        ### Return inverse covmat (w/ delta correcction)
        elif not self.alphabeta_covmat:
            self.inverse_covariance_matrix()

    def get_requirements(self):
        # State requisites to the theory code
        #reqs = {"angular_diameter_distance": {"z": self.zcmb}}
        
        ### True for cosmosis
        #if self.use_abs_mag:
            #reqs["Mb"] = None

        #return reqs
        return {}

    def _read_covmat(self, filename):

        """
        cov = np.loadtxt(filename)

        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]

        ### Reshape: self.nsn = number of SN
        return cov.reshape((self.nsn, self.nsn))
        """

        ################################ COSMOSIS ################################
        """Run once at the start to build the covariance matrix for the data"""
        #filename = self.options.get_string("covmat_file", default=default_covmat_file)
        print("Loading Pantheon covariance from {}".format(filename))
        
        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This data file is just the systematic component of the covariance - 
        # we also need to add in the statistical error on the magnitudes
        # that we loaded earlier
        
        f = open(filename)
        line = f.readline()
        n = int(len(self.zCMB)) # reading selected zcmb
        C = np.zeros((n,n))
        ii = -1
        jj = -1
        mine = 999
        maxe = -999
        for i in range(self.origlen):
            jj = -1
            if self.ww[i]:
                ii += 1
            for j in range(self.origlen):
                if self.ww[j]:
                    jj += 1
                val = float(f.readline())
                """
                if type(f.readline()) == str:
                    print("DEBUG: %d"%i)
                    print(self.origlen)
                    print("DEBUG: {}".format(f.readline()))
                    raise Exception("String!")
                """
                if self.ww[i]:
                    if self.ww[j]:
                        C[ii,jj] = val
        f.close()

        #print("DEBUG: " + str(self.use_abs_mag))

        # Return the covariance; the parent class knows to invert this
        # later to get the precision matrix that we need for the likelihood.
        return C


    def inverse_covariance_matrix(self, alpha=0, beta=0):
        ### Copy covmat with 'mag' label
        if 'mag' in self.covs:
            invcovmat = self.covs['mag'].copy()
        else:
            invcovmat = 0

        ### False
        if self.alphabeta_covmat:
            if np.isclose(alpha, self._last_alpha) and np.isclose(beta, self._last_beta):
                return self.invcov
            self._last_alpha = alpha
            self._last_beta = beta
            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            if 'stretch' in self.covs:
                invcovmat += alphasq * self.covs['stretch']
            if 'colour' in self.covs:
                invcovmat += betasq * self.covs['colour']
            if 'mag_stretch' in self.covs:
                invcovmat += 2 * alpha * self.covs['mag_stretch']
            if 'mag_colour' in self.covs:
                invcovmat -= 2 * beta * self.covs['mag_colour']
            if 'stretch_colour' in self.covs:
                invcovmat -= 2 * alphabeta * self.covs['stretch_colour']
            delta = (self.pre_vars + alphasq * self.stretch_var +
                     betasq * self.colour_var + 2.0 * alpha * self.cov_mag_stretch +
                     -2.0 * beta * self.cov_mag_colour +
                     -2.0 * alphabeta * self.cov_stretch_colour)
        
        ### Useless for cosmosis
        #else:
            #delta = self.pre_vars
        
        ### delta = diagonal part of the statistical uncertainty
        ### Removed in cosmosis
        # np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)

        ### Take inverse
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov

    def alpha_beta_logp(self, lumdists, alpha=0, beta=0, Mb=0, invcovmat=None):
        ### False
        if self.alphabeta_covmat:
            if self.use_abs_mag:
                self.log.warning("You seem to be using JLA with the absolute magnitude "
                                 "module. JLA uses a different callibration, the Mb "
                                 "module only works with Pantheon SNe!")
                estimated_scriptm = Mb
            else:
                alphasq = alpha * alpha
                betasq = beta * beta
                alphabeta = alpha * beta
                invvars = 1.0 / (self.pre_vars + alphasq * self.stretch_var +
                                 betasq * self.colour_var +
                                 2.0 * alpha * self.cov_mag_stretch -
                                 2.0 * beta * self.cov_mag_colour -
                                 2.0 * alphabeta * self.cov_stretch_colour)
                wtval = np.sum(invvars)
                estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            diffmag = (self.mag - lumdists + alpha * self.stretch -
                       beta * self.colour - estimated_scriptm)
            if invcovmat is None:
                invcovmat = self.inverse_covariance_matrix(alpha, beta)
        
        else:
            ### True for cosmosis
            if self.use_abs_mag:
                estimated_scriptm = Mb
            
            else:
                ### Inverse of systematic uncertainties
                invvars = 1.0 / self.pre_vars

                ### Sum of invvars
                wtval = np.sum(invvars)

                ### ??
                estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            
            ### Data vector? Diff between theory and observed, with correction
            ### (self.mag - estimated_scriptm) = observed/corrected mu?
            diffmag = self.mag - lumdists - estimated_scriptm
            
            ### Retreive invcov
            invcovmat = self.invcov

        ### Taking dot product to get chi^2
        invvars = invcovmat.dot(diffmag) # (covmat)^-1 dot datavector
        amarg_A = invvars.dot(diffmag) # innvars dot data vector

        ### False
        if self.twoscriptmfit:
            # could simplify this..
            amarg_B = invvars.dot(self.A1)
            amarg_C = invvars.dot(self.A2)
            invvars = invcovmat.dot(self.A1)
            amarg_D = invvars.dot(self.A2)
            amarg_E = invvars.dot(self.A1)
            invvars = invcovmat.dot(self.A2)
            amarg_F = invvars.dot(self.A2)
            tempG = amarg_F - amarg_D * amarg_D / amarg_E
            assert tempG >= 0
            if self.use_abs_mag:
                chi2 = amarg_A + np.log(amarg_E / _twopi) + np.log(tempG / _twopi)
            else:
                chi2 = (amarg_A + np.log(amarg_E / _twopi) +
                        np.log(tempG / _twopi) - amarg_C * amarg_C / tempG -
                        amarg_B * amarg_B * amarg_F / (amarg_E * tempG) +
                        2.0 * amarg_B * amarg_C * amarg_D / (amarg_E * tempG))
        ### ??
        else:
            amarg_B = np.sum(invvars)
            amarg_E = np.sum(invcovmat)

            ### True for cosmosis
            if self.use_abs_mag:
                chi2 = amarg_A + np.log(amarg_E / _twopi)
            
            ### ??
            else:
                chi2 = amarg_A + np.log(amarg_E / _twopi) - amarg_B ** 2 / amarg_E
        
        ##
        print("Cobaya Likelihood: ", -chi2/2)
        if record_like:
            file_like.write(str(-chi2/2)+ '\n')

        return - chi2 / 2

    def logp(self, **params_values):

        '''
        ### calculate theory
        angular_diameter_distances = \
            self.provider.get_angular_diameter_distance(self.zCMB)
        lumdists = (5 * np.log10((1 + self.zHEL) * (1 + self.zCMB) *
                                 angular_diameter_distances))
        '''

        # calculating cosmological distances
        H0gp = params_values['H0']
        omegamgp = params_values['Omega_m']
        Mgp = params_values['Mb']

        lumdists = []
        for  i in range(len(self.zCMB)):
            lumdists.append(da(self.zCMB[i], H0gp,omegamgp) * (1. + self.zCMB[i])*(1. + self.zHEL[i]))
        lumdists = np.array(lumdists) # Vectorize
        lumdists = 5 * np.log10(lumdists) + 25

        # replacing distances to calibration SNe with distance measurements
        lumdists[self.is_calibrator] = self.cepheid_distance[self.is_calibrator]

        ### True for cosmosis
        if self.use_abs_mag:
            Mb = params_values.get('Mb', None)
            #print(Mb)
            print("DEBUG: Sampling Mb")

        else:
            print("DEBUG: Not sampling Mb")
            Mb = 0

        ### False
        if self.marginalize:
            print("DEBUG: marginalizing")
            # Should parallelize this loop
            for i in range(self.int_points):
                self.marge_grid[i] = - self.alpha_beta_logp(
                    lumdists, self.alpha_grid[i],
                    self.beta_grid[i], Mb,
                    invcovmat=self.invcovs[i])
            grid_best = np.min(self.marge_grid)
            return - grid_best + np.log(
                np.sum(np.exp(- self.marge_grid[self.marge_grid != np.inf] + grid_best)) *
                self.step_width_alpha * self.step_width_beta)
        
        else:
            ### False
            if self.alphabeta_covmat:
                return self.alpha_beta_logp(lumdists, params_values[self.alpha_name],
                                            params_values[self.beta_name], Mb)
            ### Return -chi2/2
            else:
                return self.alpha_beta_logp(lumdists, Mb=Mb)
