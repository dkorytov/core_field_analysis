#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py
import halotools
from scipy.special import erf
import warnings
from matplotlib.patches import Circle

from halotools.empirical_models import PrebuiltHodModelFactory, NFWProfile
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import return_xyz_formatted_array, tpcf

from colossus.halo import mass_adv
from colossus.cosmology import cosmology
from halotools.mock_observables import wp

from core_model import *
from abundance import get_expected_abundance
from data_wp import *
from abundance import * 

# Define our standards

# Units are physical Mpc/h, with h=0.7. We are using physical because
# wp(rp) is measured with physical distances (observers :P )

class HODModel:
    def __init__(self, log_M_min, sigma_log_M, log_M0, alpha):
        self.log_M_min   = log_M_min
        self.sigma_log_M = sigma_log_M
        self.M0          = 10**log_M0
        self.M1          = 10**log_M1
        self.alpha       = alpha
        
    def hod_cen(self, M):
        log_M = np.log10(M)
        return 0.5*(1.0 + erf((log_M - self.log_M_min)/self.sigma_log_M))

    def hod_sat(self, M):
        sat =  ((M-self.M0)/(self.M1))**self.alpha
        sat[~np.isfinite(sat)] = 0.0
        return sat
    
    @staticmethod
    def get_central_instance(hod_cen_expected):
        rnd = np.random.uniform(size=np.shape(hod_cen_expected))
        central_cnt = np.zeros_like(hod_cen_expected, dtype=np.int)
        has_central = rnd<hod_cen_expected # if hod_cen_expected[i] ==
                                           # 1, then there should be
                                           # 100% chance to have a
                                           # central there. 
        central_cnt[has_central] = 1.0
        return central_cnt

    @staticmethod
    def get_satellite_instance(hod_sat_expected):
        satellite_cnt = np.random.poisson(lam=hod_sat_expected)
        return satellite_cnt
        
    def populate(self, M, seed = None, return_expected=False):
        """Given halo masses M, this function assigns a number of galaxies to
        each halo. 
        """
        if seed is not None:
            np.random.seed(seed)
        hod_cen_expected = self.hod_cen(M)
        hod_sat_expected = self.hod_sat(M)
        if return_expected:
            return hod_cen_expected, hod_sat_expected
        else:
            central_instance = HODModel.get_central_instance(hod_cen_expected)
            satellite_instance = HODModel.get_satellite_instance(hod_sat_expected)
            satellite_instance = satellite_instance*central_instance
            return central_instance, central_instance*satellite_instance


def get_core_model(athor=None):
    pass

def get_gao_2015_hod_model(Mr=None):
    # raise None # Can't use this with our sims
    warnings.warn("The halo definitions in this model are different than ours. Don't expect results to match. Things should only be in the correct ball park")
    # HTTP://arxiv.org/pdf/1505.07861.pdf Simulation is Multidark
    # Halos are RockStar viral SOD halos...So they have overlapping
    # halos. 
    if Mr == -21.5:
        log_M_min = 13.53
        sigma_log_M = 0.72
        log_M0 = 13.13
        log_M1 = 14.52
        alpha  = 1.14
        return HODModel(log_M_min, sigma_log_M, log_M0, log_M1, alpha)
    elif Mr == -21:
        log_M_min = 12.78
        sigma_log_M = 0.49
        log_M0 =  12.59
        log_M1 = 13.99
        alpha  = 1.14
        return HODModel(log_M_min, sigma_log_M, log_M0, log_M1, alpha)
    elif Mr == -20.5:
        log_M_min = 12.23
        sigma_log_M =0.18
        log_M0 =  12.42
        log_M1 = 13.57
        alpha  = 1.06
        return HODModel(log_M_min, sigma_log_M, log_M0, log_M1, alpha)
    elif Mr == -20:
        log_M_min = 11.95
        sigma_log_M = 0.16
        log_M0 = 12.10
        log_M1 = 13.33
        alpha  = 1.08
        return HODModel(log_M_min, sigma_log_M, log_M0, log_M1, alpha)
    elif Mr == -19.5:
        log_M_min = 11.67
        sigma_log_M = 0.01
        log_M0 = 11.80
        log_M1 = 13.07
        alpha  = 1.06
        return HODModel(log_M_min, sigma_log_M, log_M0, log_M1, alpha)

    # elif Mr == -21:
    #     log_M_min = 
    #     simga_log_M =
    #     log_M0 = 
    #     log_M1 = 
    #     alpha  = 
    else:
        raise KeyError("Mr={:.2f} is not a supported value [-21.5]")

def get_parejko_2015_hod_model():
    # The abundance should be 3e-4 h^3Mpc^-3, comoing, h=?
    log_M_cut   = 13.25
    log_M1      = 14.18
    sigma_log_M = 0.70
    kappa       = 1.04
    alpha       = 0.94
    #=======
    log_M0 = log_M_cut + np.log10(kappa)
    # ========
    log_M_min = log_M_cut
    # log_M_min = 13.53
    # sigma_log_M = 0.72
    # log_M0 = 13.13
    # log_M1 = 14.52
    # alpha  = 1.14
    return HODModel(log_M_min, sigma_log_M, log_M0, log_M1, alpha)
    
def prep_galaxy_dict(size):
    galaxies = {}
    for key in ['x', 'y', 'z', 'halo_mass']:
        galaxies[key] = np.zeros(size, dtype=np.float)
    galaxies['htag'] = np.zeros(size, dtype='i8')
    galaxies['central'] = np.zeros(size, dtype=bool)
    return galaxies

def get_spherical_position(num):
    vec = np.random.randn(3, num)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def populate_halo_with_galaxies(nfw_profile, galaxy_num, halo_mass,
                                concentration):
    x = np.zeros((galaxy_num), dtype=float)
    y = np.zeros((galaxy_num), dtype=float)
    z = np.zeros((galaxy_num), dtype=float)
    if galaxy_num > 0:
        x[0] = 0
        y[0] = 0
        z[0] = 0
    if galaxy_num > 1:
        radial_positions = nfw_profile.mc_generate_nfw_radial_positions(num_pts=galaxy_num-1, halo_mass = halo_mass, conc=concentration)
        spherical_positions = get_spherical_position(galaxy_num-1)
        x[1:] = spherical_positions[0,:]*radial_positions
        y[1:] = spherical_positions[1,:]*radial_positions
        z[1:] = spherical_positions[2,:]*radial_positions
    return x, y, z
    
def populate_halos_with_galaxies_with_NFW(fof_cat):
    fof_cat_size= len(fof_cat['x'])
    galaxy_size = np.sum(fof_cat['galaxy_number'])
    galaxy_dict = prep_galaxy_dict(galaxy_size)
    gal_offset = 0
    nfw_profile = NFWProfile(mdef='vir', conc_mass_model='dutton_maccio14')
    for i in range(0, fof_cat_size):
        if(i%100000 == 1):
            print(i, fof_cat_size, i/fof_cat_size)
        gal_num = fof_cat['galaxy_number'][i]
        if gal_num > 0:        
            gal_a, gal_b = gal_offset, gal_offset+gal_num
            x, y, z = populate_halo_with_galaxies(nfw_profile, 
                                                  gal_num,
                                                  fof_cat['m180b'][i], 
                                                  fof_cat['c180b'][i])
            # if(gal_num > 5):
            #     plt.figure()
            #     plt.plot(x, y,'o')
            #     circle = Circle((0,0), fof_cat['r180b'][i], fc='none', ec='k')
            #     plt.gca().add_artist(circle)
            #     plt.xlim([-fof_cat['r180b'][i], fof_cat['r180b'][i]])
            #     plt.ylim([-fof_cat['r180b'][i], fof_cat['r180b'][i]])
            #     print('m: ', fof_cat['r180b'][i])
            #     print('c: ', fof_cat['c180b'][i])
            #     print()
            #     plt.show()
            galaxy_dict['x'][gal_a:gal_b] = x + fof_cat['x'][i]
            galaxy_dict['y'][gal_a:gal_b] = y + fof_cat['y'][i]
            galaxy_dict['z'][gal_a:gal_b] = z + fof_cat['z'][i]
            galaxy_dict['htag'][gal_a:gal_b] = fof_cat['htag'][i]
            galaxy_dict['central'][gal_a] = 1.0
            galaxy_dict['central'][gal_a+1:gal_b] = 0.0
        gal_offset+= gal_num
    return galaxy_dict
        
def populate_halos_with_galaxies(fof_cat, method="NFW"):
    if method == "NFW":
        return populate_halos_with_galaxies_with_NFW(fof_cat)
    else:
        raise
    
def test_hod_model():
    masses = np.logspace(11, 15, 100)
    f, ax = plt.subplots()
    for Mr in [-21.5, -21.0, -20.5, -20, -19.5]:
        hod_model = get_gao_2015_hod_model(Mr=Mr)
        cen, sat = hod_model.populate(masses, return_expected=True)
        ax.loglog(masses, cen+sat, '-', label='Mr<{:.1f}'.format(Mr))
    ax.set_xlim([1e11, 1e15])
    ax.set_ylim([0.1, 100])
    ax.legend(loc='best')
    plt.show()

def apply_hod_to_fof():
    Mr = -20
    fof_cat = load_fof_cat()
    hod_model = get_gao_2015(Mr=Mr)
    fof_cen, fof_sat = hod_model.populate(fof_cat['m180b'])
    fof_cat['central_galaxies_number'] = fof_cen
    fof_cat['satellite_galaxies_number'] = fof_sat
    fof_cat['galaxy_number'] = fof_cen + fof_sat
    gals = populate_halos_with_galaxies(fof_cat)
    
    gal_num = len(gals['x'])
    expected_abundance = get_expected_abundance(Mr)
    hod_abundance = np.float(gal_num)/(256.0**3)
    print("expected abundance: :", expected_abundance)
    print("HOD abundance: ", hod_abundance)
    print("hod/lum func abudance: ", hod_abundance/expected_abundance)
    plt.figure()
    plt.plot(gals['x'], gals['y'], '.', alpha=0.3)

    gal_xyz = return_xyz_formatted_array(gals['x'], gals['y'], gals['z'])
    r_bins_edges = np.logspace(-1, 1, 32)
    r_bins_cen = dtk.bins_avg(r_bins_edges)


    r_bins_cen2 = (r_bins_edges[1:]+r_bins_edges[:-1])/2.0
    xi = wp(sample1=gal_xyz, rp_bins=r_bins_edges, pi_max=50.0, period=256.0)
    print(gal_xyz.shape)
    print(xi.shape)
    print(r_bins_edges.shape)
    
    print("===")
    print(r_bins_edges.shape)
    print(xi.shape)
    plt.figure()
    plt.loglog(r_bins_cen, xi, label='HOD default')
    plt.show()
    
def generate_hod_model():
    apply_hod_to_fof()

if __name__ == "__main__":
    generate_hod_model()
