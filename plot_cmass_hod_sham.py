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

from cmass_hod_sham import create_cmass_hod
from colossus.halo import mass_adv
from colossus.cosmology import cosmology


def convert_step401_fofmass_to_m180b(fof_mass, return_all=False):
    # for step 323, it's 0.92 so it's close enough
    m200c = fof_mass*0.9 #from calc_fof_sod_conversion.py for step 401
    cosmology.setCosmology('WMAP7')

    m180b, r180b, c180b = mass_adv.changeMassDefinitionCModel(m200c/0.7, 0.2, 
                                               "200c", "180m", 
                                               c_model='child18')
    m180b *=0.7
    if return_all:
        return m180b, r180b, c180b
    else:
        return m180b


def plot_cmass_hod_sham(fname_sham = "tmp/infall_hod.cmass.AQ.hdf5"):
    sham = dtk.load_dict_hdf5(fname_sham)
    cmass_hod = create_cmass_hod()
    masses = np.logspace(12, 16, 100)
    masses_cen = dtk.bins_cen(masses)

    hod_cen, hod_sat = cmass_hod.populate(masses, return_expected=True)
    hod_tot = hod_cen + hod_sat
    
    m180b = convert_step401_fofmass_to_m180b(sham['mass_bins_cen'])
    print(sham.keys())
    plt.figure()
    plt.loglog(m180b, sham['avg_cnt'], '-o', label=r'$\rm{OR-CMASS~SHAM}$')
    plt.loglog(masses, hod_tot, '-', label=r'$\rm{CMASS HOD}$')
    plt.legend(loc='best')
    plt.ylabel(r'$\rm{Mean~Halo~Occupancy}$')
    plt.xlabel(r'$\rm{M_{180b}~[h^{-1} M_\odot},~h=0.7]$')
    plt.show()

if __name__ == "__main__":
    print(colossus.__file__)
    plot_cmass_hod_sham('tmp/infall_hod.cmass.OR.hdf5')
