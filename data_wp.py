
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py
import halotools

from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import return_xyz_formatted_array, tpcf
from colossus.halo import mass_adv
from colossus.cosmology import cosmology
from halotools.mock_observables import wp

from abundance import get_expected_abundance

def get_mstar():
    return -21.22

def get_lstar():
    return 2.25e10 #[Ldot] h=?

def Ldot_from_mstar(offset=0):
    return 10.352+offset/-2.5

def get_gao_2pt_mags():
   return np.array([-21.6, -21.8, -22.0])
   
def get_gao_2pt(Mi=-21.6):
    """ from https://arxiv.org/pdf/1401.3009.pdf 
    They have h=0.7
    """
    
    if Mi == -21.6:
        wp = np.array([10029.85, 4744.7, 2860.37, 2732.31, 1560.89, 1025.62, 629.37, 363.88, 195.87, 128.8, 92.67, 67.65, 48.11, 32.13, 19.56, 10.59, 3.73, 1.24])
        wp_err = np.array([7420.09, 883.75, 359.76, 236.19, 105.03, 60.78, 30.65, 12.43, 6.92, 4.58, 2.99, 2.09, 1.55, 1.33, 1.13, 0.87, 0.62, 0.5])
    elif Mi == -21.8:
        wp = np.array([ 18138.15, 6285.53, 4659.69, 3968.58, 1774.97, 1211.53, 830.34, 441.23, 233.84, 154.92, 106.21, 80.95, 55.99, 36.70, 22.27, 12.8, 4.58, 2.03])
        wp_err = np.array([ 11600.27, 1823.47, 769.44, 20.99, 171.64, 108.33, 54.38, 19.88, 12.04, 6.84, 4.35, 3.03, 2.47, 1.78, 1.43, 1.12, 0.85, 0.66])
    elif Mi == -22.0:
        wp = np.array([6986.76, 12368.79, 5602.46, 5285.31, 2225.19, 1792.21, 1009.84, 620.87, 333.99, 199.46, 124.62, 101.29, 65.71, 44.27, 24.94, 15.17, 5.82, 2.64])
        wp_err = np.array([5854.06, 4577.34, 1270.36, 911.13, 63.07, 245.61, 106.53, 43.63, 23.18, 14.81, 7.34, 5.62, 4.36, 2.58, 2.11, 1.54, 1.21, 0.91]) 
    else:
        print("Isn't an value given by Gao+2012 [-21.6, -21.8, -22.0]. Mi given is {}".format(Mi))
        raise KeyError
    return wp, wp_err

def get_gao_r_bins_centers():
    # r_bins = np.logspace(-1.77, 1.8, 19)
    r_bins = np.array([0.021, 0.033, 0.052, 0.082, 0.129, 0.205, 0.325, 0.515, 0.815, 1.292, 2.048, 3.246, 5.145, 8.155, 12.92, 20.48, 32.46, 51.45])
    print(r_bins)
    print(r_bins.shape)
    return r_bins

def get_gao_r_bins_edges():
    return decenter_log_bins(get_gao_r_bins_centers())

def decenter_log_bins(bins_cen):
    lg_bins_cen = np.log(bins_cen)
    spacing = np.mean(np.diff(lg_bins_cen))
    bins_edges = np.zeros(len(bins_cen)+1)
    bins_edges[:-1] = lg_bins_cen -spacing/2.0
    bins_edges[-1] = lg_bins_cen[-1]+spacing/2.0
    return np.exp(bins_edges)
    
def get_cac09_md_hod(mass_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='multidark', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    h_gal,  _  = np.histogram(model.mock.galaxy_table['halo_mvir'], bins=mass_bins)
    h_halo, _  = np.histogram(halocat.halo_table['halo_mvir'],      bins=mass_bins)
    return h_gal/h_halo

def get_cac09_bp_hod(mass_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='bolplanck', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    print(halocat.halo_table.keys())
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    h_gal,  _  = np.histogram(model.mock.galaxy_table['halo_mvir'], bins=mass_bins)
    h_halo, _  = np.histogram(halocat.halo_table['halo_mvir'],      bins=mass_bins)
    h = h_gal/h_halo
    print("HOD: ", h)
    print("h_gal:", h_gal)
    print("h_halo:", h_halo)
    return h_gal/h_halo

def get_cac09_md_2pt(r_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='multidark', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    print('starting 2pt')
    # xi = tpcf(pos, r_bin_edges, period=halocat.Lbox)
    print("=====")
    xi = wp(pos, r_bins, pi_max=50, period=halocat.Lbox, num_threads='max')
    return xi

def get_cac09_bp_2pt(r_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='bolplanck', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x']*0.7, 
                                     model.mock.galaxy_table['y']*0.7, 
                                     model.mock.galaxy_table['z']*0.7, )
    print('starting 2pt')
    # xi = tpcf(pos, r_bin_edges, period=halocat.Lbox)
    xi = wp(pos, r_bins, pi_max=50, period=halocat.Lbox, num_threads='max')
    print("wp(r): ", xi)
    plt.figure()
    plt.plot(r_bins[:-1], xi)
    return xi

def get_parejko_r_bins_centers():
    r_bins_center = np.array([0.385, 0.577, 0.865, 1.299, 19.45, 2.921, 4.381, 6.572, 9.858, 14.78, 22.18, 33.27])
    return r_bins_center

def get_parejko_r_bins_edges():
    return decenter_log_bins(get_parejko_r_bins_centers())

def get_parejko_2pt():
    # what are the full, NGC and SGC samples?
    # North/south galactic caps. Just different survey areas
    wprp =     np.array([619.29, 390.51, 242.17, 153.87, 106.83, 84.18, 60.42, 43.19, 30.70, 19.83, 12.66, 7.22])
    wprp_err = np.array([27.20, 18.80, 9.51, 5.60, 4.60, 3.17, 2.30, 1.76, 1.69, 1.36, 0.85, 0.87])
    return wprp, wprp_err
