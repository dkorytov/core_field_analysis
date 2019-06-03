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


from observation_data_and_fits import *
from util import *

def plot_cmass(ax0, ax1):
    cmass_r_edges, cmass_r_cen  = get_cmass_r_bins()
    cmass_wp, cmass_wp_err = get_cmass_wp()
    ax0.loglog(cmass_r_cen, cmass_wp, label=r'$\rm{CMASS~Obs.}$', color='k', lw=2)
    ax0.fill_between(cmass_r_cen, cmass_wp-cmass_wp_err, cmass_wp+cmass_wp_err,color='k', alpha=0.3)

    ax1.fill_between(cmass_r_cen, -cmass_wp_err/(cmass_wp), cmass_wp_err/(cmass_wp), color='k', alpha=0.3)
    ax1.axhline(0, ls='--', color='k')
    ax1.set_ylim([-1,1])
    ax1.set_xscale('log')

def plot_wp(ax0, ax1, rp, wp, label, color):
    cmass_r_edges, cmass_r_cen  = get_cmass_r_bins()
    cmass_wp, cmass_wp_err = get_cmass_wp()

    ax0.loglog(rp, wp, color=color, label=label)
    
    x, y_diff, y_diff_relative = dtk.diff_curves(rp, wp, cmass_r_cen, cmass_wp, log=True)
    ax1.plot(x, y_diff_relative, color=color)

def plot_hod_nfw(ax0, ax1, color='b', label=r'$\rm{OR~HOD~NFW}$', fname='cache/wps/cmass_hod_nfw.hdf5'):
    dic = load_wp(fname)
    plot_wp(ax0, ax1, dic['rp'], dic['wp'], color=color, label=label)
    return 

def plot_sham_nfw(ax0, ax1, color='g', label=r'$\rm{AQ~SHAM~NFW}$', fname='cache/wps/cmass_sham_nfw.hdf5'):
    dic = load_wp(fname)
    plot_wp(ax0, ax1, dic['rp'], dic['wp'], color=color, label=label)

def plot_core_am(ax0, ax1, color='r', label=r'$\rm{OR~Core~AM}$', fname='cache/wps/cmass_core_am.hdf5'):
    dic = load_wp(fname)
    plot_wp(ax0, ax1, dic['rp'], dic['wp'], color=color, label=label)

def plot_core_fit_hard(ax0, ax1, color='c', label=r'$\rm{OR~Core~Fit}$', fname='cache/wps/cmass_core_fit.hdf5'):
    dic = load_wp(fname)
    plot_wp(ax0, ax1, dic['rp'], dic['wp'], color=color, label=label)
    

def plot_all_cmass_wp():
    plt.figure()
    ax0 = plt.subplot2grid((8,1), (0,0), 5, 1)
    ax1 = plt.subplot2grid((8,1), (5,0), 3, 1)
    
    plot_cmass(ax0, ax1)
    plot_hod_nfw(ax0, ax1)
    plot_sham_nfw(ax0, ax1)
    plot_core_am(ax0, ax1)
    plot_core_fit_hard(ax0, ax1)

    ax0.legend(loc='best')
    xlim = ax1.get_xlim()
    ax0.set_xticklabels([])
    ax1.set_xlim(xlim)
    plt.show()



if __name__ == "__main__":
    plot_all_cmass_wp()
