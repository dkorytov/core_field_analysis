#!/usr/bin/env python2.7

from __future__ import print_function, division
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec
import dtk
import h5py
import time
import halotools

from scipy.special import erfc

from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import return_xyz_formatted_array, tpcf
from colossus.halo import mass_adv
from colossus.cosmology import cosmology
from halotools.mock_observables import wp


from generate_hod_models import HODModel, populate_halos_with_galaxies_with_NFW
# from calc_wp import 

class HODModelCMASS(HODModel):
    def __init__(self, lgMcut, lgM1, sigma, kappa, alpha):
        self.Mcut   = 10**lgMcut
        self.M1     = 10**lgM1
        self.sigma2sqrt  = sigma*np.sqrt(2)
        self.kappa  = kappa
        self.alpha  = alpha
    
    def hod_cen(self, m180b):
        return 0.5*erfc(np.log(self.Mcut/m180b)/(self.sigma2sqrt))
    
    def hod_sat(self, m180b):
        sats = ( (m180b-self.kappa*self.Mcut)/self.M1 )**self.alpha
        sats[~np.isfinite(sats)] = 0.0
        return sats

def create_cmass_hod():
    lgMcut = 13.04
    lgM1   = 14.05
    sigma  = 0.94
    kappa  = 0.93
    alpha  = 0.97
    return HODModelCMASS(lgMcut, lgM1, sigma, kappa, alpha)

def load_sod_cat(fname, redshift):
    print("loading sod cat")
    t1 = time.time()
    cat = {}
    # dtk.gio_inspect(fname)
    cat['htag'] = dtk.gio_read(fname, 'fof_halo_tag')
    cat['x'] = dtk.gio_read(fname, 'fof_halo_center_x')
    cat['y'] = dtk.gio_read(fname, 'fof_halo_center_y')
    cat['z'] = dtk.gio_read(fname, 'fof_halo_center_z')
    cat['m200c'] = dtk.gio_read(fname, 'sod_halo_mass') #m200c, Msun/h, h=0.7
    cat['m180b'], cat['r180b'], cat['c180b'] = mass_adv.changeMassDefinitionCModel(cat['m200c']/0.7, redshift, 
                                                                                   "200c", "180m", 
                                                                                   c_model='child18')

    cat['m180b'] = cat['m180b']*0.7 #Msun/h, h=0.7
    cat['r180b'] = cat['r180b']*0.7/1000.0 #kpc/h, h=1 -> mpc/h, h=0.7
    t2 = time.time()
    print("\ttime: ", t2-t1)
    return cat
    

def generate_cmass_hod_galaxies(sod_cat):
    cmass_hod = create_cmass_hod()
    slct = (sod_cat['x'] < 500) & (sod_cat['y'] < 500) & (sod_cat['z'] < 500)
    sod_cat = dtk.select_dict(sod_cat, slct)
    sod_cen, sod_sat = cmass_hod.populate(sod_cat['m180b'])
    sod_tot = sod_cen + sod_sat
    sod_cat['galaxy_number'] = sod_tot
    gal_cat = populate_halos_with_galaxies_with_NFW(sod_cat)
    dtk.save_dict_hdf5('cache/cmass_hod_gal.small.hdf5', gal_cat)
    plt.figure()
    h, xbins, ybins = np.histogram2d(gal_cat['x'], gal_cat['y'], bins=256)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.show()

def load_cmass_hod_galaxies(fname='cache/cmass_hod_gal.small.hdf5'):
    return dtk.load_dict_hdf5(fname)

def get_cmass_r_bins():
    r_bins_cen = np.array([0.40, 0.71, 1.27, 2.25, 4.00, 7.11, 12.65, 22.50])
    r_bins_edges = dtk.decenter_log_bins(r_bins_cen)
    return r_bins_edges, r_bins_cen

def get_cmass_rwp():
    rwp = np.array([167.68, 134.49, 147.64, 168.29, 208.77, 242.70, 255.89, 230.36])
    err = np.array([ 15.91,   6.26,   6.54,   7.87,   9.89,  14.16 , 20.49,  28.77])
    return rwp, err

def get_cmass_wp():
    rwp, rerr = get_cmass_rwp()
    r_bin_edges, r_bin_cen = get_cmass_r_bins()
    return rwp/r_bin_cen, rerr/r_bin_cen

def compute_cmass_hod_2pt(gal_dict):
    core_xyz = return_xyz_formatted_array(gal_dict['x'],
                                          gal_dict['y'],
                                          gal_dict['z'])
    r_bin_edges = np.logspace(-1,1.5,100)
    r_bin_cen   = dtk.bins_avg(r_bin_edges)
    xi = wp(core_xyz, r_bin_edges, pi_max=50, period=500)
    gs = gridspec.GridSpec(4,4)
    plt.figure()
    # f, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]})
    ax0 = plt.subplot(gs[:3,:])
    ax0.loglog(r_bin_cen, xi, label=r'$\rm{OuterRim~HOD}$')
    cmass_r_edges, cmass_r_cen  = get_cmass_r_bins()
    cmass_wp, cmass_wp_err = get_cmass_wp()
    ax0.loglog(cmass_r_cen*0.7, cmass_wp, label=r'$\rm{CMASS~Data}$')
    ax0.fill_between(cmass_r_cen*0.7, cmass_wp-cmass_wp_err, cmass_wp+cmass_wp_err,color='g', alpha=0.3)
    ax0.legend(loc='best')
    ax0.set_ylabel(r'$\omega_p(r_p)$')


    ax1 = plt.subplot(gs[-1,:])
    x,y_diff, y_diff_relative = dtk.diff_curves(r_bin_cen, xi, cmass_r_cen*0.7, cmass_wp)
    ax1.plot(x, y_diff_relative)
    ax1.axhline(0, ls='--', color='k')
    ax1.fill_between(cmass_r_cen*0.7, -cmass_wp_err/(cmass_wp), cmass_wp_err/(cmass_wp), color='g', alpha=0.3)
    ax1.set_ylabel(r'$\frac{HOD-Data}{Data}$')
    ax1.set_ylim([-1,1])
    ax1.set_xlabel(r'$r_p~[h^{-1}Mpc, h=0.7]$')
    ax1.set_xscale('log')
    ax1.set_xlim(ax0.get_xlim())
    ax0.set_xticklabels([])    

def calc_cmass_hod_2pt():
    gal_dict = load_cmass_hod_galaxies()
    plt.figure()
    plt.plot(gal_dict['x'], gal_dict['y'], ',', alpha=1.0,)
    rl = 500
    vol = rl*rl*rl
    num = gal_dict['x'].size
    abundance = num/vol
    print("abundance: {:.2e}".format(abundance))
    print("abundance relative error: {:.2f}".format((abundance-3.6e-4)/3.6e-4))
    plt.show()

    compute_cmass_hod_2pt(gal_dict)

if __name__ == "__main__":
    # calc_cmass_hod_2pt()    
    # plt.show()
    # exit()

    print("yo")
    cosmology.setCosmology('WMAP7')
    cmass_hod = create_cmass_hod()
    stepz = dtk.StepZ(sim_name='AlphaQ')
    step = 323
    step_z = stepz.get_z(step)
    sod_gio_fname = '/media/luna1/dkorytov/data/OuterRim/sod/m000.${step}.sodproperties'
    sod_hdf5_fname = "cache/sod180b.${step}.hdf5"
    # sod_cat = load_sod_cat(sod_gio_fname.replace("${step}", str(step)), step_z)
    # dtk.save_dict_hdf5(sod_hdf5_fname.replace("${step}", str(step)), sod_cat)
    sod_cat = dtk.load_dict_hdf5(sod_hdf5_fname.replace("${step}", str(step)))
    generate_cmass_hod_galaxies(sod_cat)
    sod_cen, sod_sat = cmass_hod.populate(sod_cat['m180b'], return_expected=True)
    sod_tot = sod_cen+sod_cen*sod_sat

    masses = np.logspace(10,16, 100)
    cen = cmass_hod.hod_cen(masses)
    sats = cmass_hod.hod_sat(masses)
    tot = cen + cen*sats
    plt.figure()
    plt.loglog(masses, tot, '-b')
    plt.plot(masses, cen, '--b')
    plt.plot(masses, sats, ':b')
    print(tot.size)
    hod_gal_avg  = dtk.binned_average(sod_cat['m180b'], sod_tot, masses)
    masses_cen = dtk.log_bins_avg(masses)
    print(hod_gal_avg.size)
    print(masses_cen.size)
    plt.plot(masses_cen, hod_gal_avg, '-r')
    
    plt.figure()
    h,_ = np.histogram(sod_cat['m180b'], bins=masses)
    plt.loglog(masses_cen, h)

    plt.figure()
    plt.semilogx(masses_cen, h*hod_gal_avg)

    sod_pop_cen, sod_pop_sat = cmass_hod.populate(sod_cat['m180b'])
    print(sod_pop_cen, sod_pop_sat)
    sod_pop_tot = sod_pop_cen + sod_pop_cen*sod_pop_sat

    calc_cmass_hod_2pt()    

    plt.show()
