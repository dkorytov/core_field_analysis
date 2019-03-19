

from __future__ import print_function, division
import numpy as np
import Corrfunc.mocks as cf

def get_core_ra_dec(core_xyz):
    CZ = np.sqrt(core_xyz[:,0]**2  + core_xyz[:,1]**2  + core_xyz[:,2]**2)
    inv_cz = 1.0/CZ
    x = core_xyz[:,0]*inv_cz
    y = core_xyz[:,1]*inv_cz
    z = core_xyz[:,2]*inv_cz
    DEC = 90.0 - np.arccos(z)*180/np.pi
    RA = (np.arctan2(y, x)*180/np.pi)+180.0
    return RA, DEC, CZ
    
def calc_corrfunc_wp(rL, r_bins, RA, DEC, CZ, weights=None, pi_max = 50.0, nthreads=12):
    autocorr = 1
    cosmology =2 #Planck
    results = cf.DDrppi_mocks(autocorr, cosmology, nthreads,
                              pi_max, r_bins, 
                              RA, DEC, CZ, 
                              weights1=weights,
                              weight_type='pair_product',
                              is_comoving_dist=True)
    print(results)
    return results

