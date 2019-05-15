#!/usr/bin/env python2.7

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

from core_model import *
from abundance import get_expected_abundance
from calc_wp import *
from data_wp import *

def decenter_log_bins(bins_cen):
    lg_bins_cen = np.log(bins_cen)
    spacing = np.mean(np.diff(lg_bins_cen))
    bins_edges = np.zeros(len(bins_cen)+1)
    bins_edges[:-1] = lg_bins_cen -spacing/2.0
    bins_edges[-1] = lg_bins_cen[-1]+spacing/2.0
    return np.exp(bins_edges)
    
    

def compare_wp_targets():
    gao_r_bins_cen = get_gao_r_bins()
    gao_r_bins_edges = decenter_log_bins(gao_r_bins_cen)
    colors = ['b', 'g', 'r', 'm']
    mstar_offsets = []
    print(gao_r_bins_cen)
    f, axs = plt.subplots(1,3)
    for i, mag in enumerate(get_gao_2pt_mags()):
        gao_wp, gao_wp_error = get_gao_2pt(mag)
        mstar_offsets.append(get_mstar()-mag)
        c = colors[i]
        axs[i].loglog(gao_r_bins_cen, gao_wp, color=c, label='Gao+12, Mr={:.1f}'.format(mag))
        axs[i].fill_between(gao_r_bins_cen, gao_wp+gao_wp_error, gao_wp-gao_wp_error, color=c, alpha=0.2)
    for i, mstar_offset in enumerate(mstar_offsets):
        ldot = Ldot_from_mstar(offset=mstar_offset)
        print(mstar_offset,"->", ldot)
        wp = get_cac09_md_2pt(gao_r_bins_edges, threshold=ldot)
        axs[i].loglog(gao_r_bins_cen, wp, label='Caco+09 MD Lsun={:.1f}'.format(ldot), color=colors[i])
        wp = get_cac09_bp_2pt(gao_r_bins_edges, threshold=ldot)
        axs[i].loglog(gao_r_bins_cen, wp, '--', label='Caco+09 BP Lsun={:.1f}'.format(ldot), color=colors[i])

if __name__ == "__main__":
    compare_wp_targets()
    plt.show()
