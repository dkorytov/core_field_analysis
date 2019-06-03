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

from util import *
from plot_cmass_hod_sham import convert_step401_fofmass_to_m180b
from generate_hod_models import populate_halos_with_galaxies_with_NFW
from cmass_hod_sham import compute_cmass_hod_2pt
from cmass_hod_sham import get_cmass_wp, get_cmass_r_bins


def calc_core_sham(abundance, rl):
    fname = "/media/luna1/dkorytov/data/OuterRim/cores_500L/03_31_2018.OR.401.coreproperties"
    cat = load_core_cat(fname)
    expected_count = abundance*rl**3
    mass_cut = get_infall_by_count(cat, expected_count)
    cat = select_dict(cat, cat['mass']>mass_cut)
    print("{:.2e}".format(mass_cut))
    r_bins = np.logspace(-1,1.5,100)
    r_bins_cen = dtk.bins_avg(r_bins)
    xi = calc_wp(cat, r_bins, 500)
    cmass_r_bins, cmass_r_bins_cen = get_cmass_r_bins()
    cmass_wp, cmass_wp_err = get_cmass_wp()
    print(np.shape(cmass_r_bins_cen))
    print(np.shape(cmass_wp))
    plt.figure()
    plt.loglog(r_bins_cen, xi, label='Core SHAM')
    plt.loglog(cmass_r_bins_cen, cmass_wp, label='CMASS Obs.')
    plt.xlabel('rp')
    plt.ylabel('wp(rp)')
    plt.legend(loc='best')
    plt.show()
    
def get_infall_by_count(cat, count):
    count = np.int(count)
    mass = np.sort(cat['mass'])
    return mass[-count]
    

if __name__ == "__main__":
    calc_core_sham(3.6e-4, 500)
