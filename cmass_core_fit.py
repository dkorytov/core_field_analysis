#!/usr/bin/env python2.7

from __future__ import print_function, division
import matplotlib
import os
#checks if there is a display to use.
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

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
import multiprocessing 

from observation_data_and_fits import *
from util import *


def pool_calc_wp(arg):
    core_model_cat =     arg[0]
    r_bins_edges = arg[1]
    rL =           arg[2]
    core_wp = calc_wp(core_model_cat, r_bins_edges, rL)
    print("...")
    return core_wp, core_model_cat['x'].size/(rL**3)

def get_parameter_sweep():
    mass_cuts = np.logspace(12, 13, 64)
    radius_cuts = np.logspace(-3, -0.5, 64)
    cost = np.zeros((len(mass_cuts), len(radius_cuts)), dtype=np.float)
    return mass_cuts, radius_cuts, cost


def plot_cost(mass_cuts, radius_cuts, cost_mat):
    plt.figure()
    plt.pcolor(mass_cuts, radius_cuts, cost_mat.T, norm=clr.LogNorm(), cmap='nipy_spectral_r')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Radius cut [ Mpc/h]')
    plt.xlabel('Mass cut [Msun/h]')

def save_wp_best_model(core_cat, rL, mass_cut, radius_cut, fname):
    print('makeing best model')
    core_model_cat = apply_core_hard_model(core_cat, mass_cut, radius_cut)
    r_bins_edges = np.logspace(-1,1.5,64)
    r_bins_cen = dtk.bins_avg(r_bins_edges)
    xi = calc_wp(core_model_cat, r_bins_edges, rL)
    result = {"wp": xi,
              "rp": r_bins_cen,}
    dtk.save_dict_hdf5(fname, result)
    

def cmass_core_fit():
    core_cat = load_OR_cores_500L()
    rL = 500
    r_bins_edges, r_bins_cen = get_cmass_r_bins()
    cmass_wp, cmass_wp_err = get_cmass_wp()
    mass_cuts, radius_cuts, cost_mat = get_parameter_sweep()
    index_converter = dtk.IndexConverter([len(mass_cuts), len(radius_cuts)])
    pool = multiprocessing.Pool(processes = 24)
    pool_args = []
    best_param = None
    best_cost = 1e10
    best_wp  = None
    for m_i, mass_cut in enumerate(mass_cuts):
        for r_i, radius_cut in enumerate(radius_cuts):
            print(m_i, r_i)
            core_model_cat = apply_core_hard_model(core_cat, mass_cut, radius_cut)
            pool_args.append([core_model_cat, r_bins_edges, rL])
            # core_wp = calc_wp(core_model_cat, r_bins_edges, rL)
            # cost = calc_wp_difference(cmass_wp, core_wp)
            # cost_mat[m_i, r_i] = cost
    results = pool.map(pool_calc_wp, pool_args)
    index = 0
    for m_i, mass_cut in enumerate(mass_cuts):
        for r_i, radius_cut in enumerate(radius_cuts):
            print(m_i, r_i)
            core_wp, core_abundance = results[index]
            cost1 = calc_wp_difference(cmass_wp, core_wp)
            cost2 = calc_abudance_difference(core_abundance, 3.6e-4)
            cost = cost1+cost2*10
            cost_mat[m_i, r_i] = cost
            index +=1
            if cost < best_cost and cost != 0:
                best_cost = cost
                best_param = [mass_cut, radius_cut]
                best_wp = core_wp
    print(cost_mat)
    plot_cost(mass_cuts, radius_cuts, cost_mat)
    plt.figure()
    plt.loglog(r_bins_cen, cmass_wp, label='cmass')
    plt.loglog(r_bins_cen, best_wp,  label='core')



    best_mass_cut = best_param[0]
    best_radius_cut = best_param[1]
    save_wp_best_model(core_cat, rL, best_mass_cut, best_radius_cut, 'cache/wps/cmass_core_fit.hdf5')
    plt.show()
    
if __name__ == "__main__":
    cmass_core_fit()
    
