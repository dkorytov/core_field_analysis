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
import colossus


def load_fof(fname):
    cat = {}
    cat['htag'] = dtk.gio_read(fname, 'fof_halo_tag')
    cat['fof_mass'] = dtk.gio_read(fname, 'fof_halo_mass')
    return cat

def load_sod(fname):
    cat = {}
    cat['htag'] = dtk.gio_read(fname, 'fof_halo_tag')
    cat['sod_mass'] = dtk.gio_read(fname, 'sod_halo_mass')
    return cat

def combine_cat(fof_cat, sod_cat):
    srt = np.argsort(fof_cat['htag'])
    indx = dtk.search_sorted(fof_cat['htag'], sod_cat['htag'], sorter=srt)
    print("No matches:", np.sum(indx==-1))
    sod_cat['fof_mass'] = fof_cat['fof_mass'][indx]
    return sod_cat
    

def plot_relation(cat):
    plt.figure()
    plt.hist2d(np.log10(cat['fof_mass']), np.log10(cat['sod_mass']), bins=100, norm=clr.LogNorm(), cmap='Blues')

    plt.figure()
    ratio = cat['sod_mass']/cat['fof_mass']
    x = np.log10(cat['fof_mass'])
    x_bins = dtk.get_logbins(x)
    x_bins_cen = dtk.log_bins_cen(x_bins)

    plt.hist2d(np.log10(cat['fof_mass']), cat['sod_mass']/cat['fof_mass'], bins=100, norm=clr.LogNorm(), cmap='Blues')
    avg = dtk.binned_average(x, ratio, x_bins)
    plt.plot(x_bins_cen, avg, '-r')

    plt.show()
    
if __name__ == "__main__":
    fof_fname = "/media/luna1/dkorytov/data/AlphaQ/fof/m000-${step}.fofproperties"
    sod_fname = "/media/luna1/dkorytov/data/AlphaQ/sod/m000-${step}.sodproperties"
    fof_cat = load_fof(fof_fname.replace("${step}", str(323)))
    sod_cat = load_sod(sod_fname.replace("${step}", str(323)))
                       
    both_cat = combine_cat(fof_cat, sod_cat)
    plot_relation(both_cat)
