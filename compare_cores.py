#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py
import halotools


def load_cores(fname):
    print("Loading ", fname)
    cat = {}
    cat['mass'] = dtk.gio_read(fname, "infall_mass")
    cat['radius'] = dtk.gio_read(fname, 'radius')
    print("Done")
    return cat

def get_counts_by_mass(cat, limits):
    result = np.zeros_like(limits)
    for i, limit in enumerate(limits):
        slct = cat['mass']>limit
        result[i] = np.sum(slct)
    return result

def get_difference(x1, x2):
    mean = (x1+x2)/2.0
    diff = x1-x2
    return diff/mean

def compare_cores(fname1, label1, rl1, fname2, label2, rl2):
    cat1 = load_cores(fname1)
    cat2 = load_cores(fname2)
    vol1 = rl1**3
    vol2 = rl2**3
    masses = np.logspace(9, 15.5, 100)
    density1 = get_counts_by_mass(cat1, masses)/vol1
    density2 = get_counts_by_mass(cat2, masses)/vol2
    
    f, ax = plt.subplots(2,1, sharex=True)
    
    ax[0].loglog(masses, density1, label=label1)
    ax[0].loglog(masses, density2, label=label2)
    ax[0].legend()
    ax[0].set_ylabel('Abundance [h^3 Mpc^-3]')
    ax[1].semilogx(masses, get_difference(density2, density1))
    ax[1].set_ylabel('QC-OR/mean(QC,OR)')
    ax[1].set_xlabel('Infall Mass [Msun/h]')
    ax[1].axhline(0.0, ls='--', color='k')
    ax[1].set_ylim([-0.2, +0.2])


def compare_500L_cores():
    fname1 = "/media/luna1/dkorytov/data/OuterRim/cores_500L/03_31_2018.OR.401.coreproperties"
    label1 = "Outer Rim"
    fname2 = "/media/luna1/dkorytov/data/QContinuum/cores_500L/03_29_17.Qcontinuum.401.coreproperties"
    label2 = "QContinuum"
    compare_cores(fname1, label1, 500, fname2, label2, 500)

def compare_full_cores():
    fname1 = "/media/luna1/dkorytov/data/OuterRim/cores/03_31_2018.OR.401.coreproperties"
    label1 = "Outer Rim"
    fname2 = "/media/luna1/dkorytov/data/QContinuum/cores/03_29_17.Qcontinuum.401.coreproperties"
    label2 = "QContinuum"
    compare_cores(fname1, label1, 923, fname2, label2, 3000)


if __name__ == "__main__":
    compare_full_cores()
    plt.show()
