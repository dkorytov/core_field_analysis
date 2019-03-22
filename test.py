#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from calc_wp import *
from colossus.cosmology import cosmology
from halotools_tpcf_interface import *

# def halotools_wtpcf(xyz, r_bins, period, weights=None):
#     wcnts = marked_npairs_3d(xyz, xyz, r_bins, period=period, weights1=weights, weights2=weights, weight_func_id=1)
#     wcnts_bins = wcnts[1:]-wcnts[:-1]
#     r_bins_vol = 4.0/3.0*np.pi*r_bins**3
#     r_bins_shell_vol = r_bins_vol[1:]-r_bins_vol[:-1]
#     if weights is None:
#         Npts = np.shape(xyz)[0]
#     else:
#         Npts = np.sum(weights)
#     expected_density = Npts/period**3
#     expected_pair_wcnts = Npts*r_bins_shell_vol*expected_density
#     return wcnts_bins/expected_pair_wcnts - 1.0

# def halotools_tpcf(xyz, r_bins, period):
#     wcnts =npairs_3d(xyz, xyz, r_bins, period=period)
#     wcnts_bins = wcnts[1:]-wcnts[:-1]
#     r_bins_vol = 4.0/3.0*np.pi*r_bins**3
#     r_bins_shell_vol = r_bins_vol[1:]-r_bins_vol[:-1]
#     Npts = np.shape(xyz)[0]
#     expected_density = Npts/period**3
#     expected_pair_wcnts = Npts*r_bins_shell_vol*expected_density
#     return wcnts_bins/expected_pair_wcnts - 1.0


if __name__ == "__main__":
    Npts = 3000
    np.random.seed(12345)
    x=np.random.normal(size=Npts)*2 + 50
    y=np.random.normal(size=Npts)*2 + 50
    z=np.random.normal(size=Npts)*2 + 50
    rL = 100
    xyz = np.vstack((x,y,z)).T
    weights = np.ones_like(x)
    r_bins = np.logspace(-1,0.5,16)
    r_bins_cen = dtk.bins_avg(r_bins)
    print("calculating 2pt")
    tpcf_ht = tpcf(xyz, r_bins, period = rL)
    # wtpcf_ht = marked_tpcf(xyz, r_bins, period = rL, weights=weights)
    wtpcf_mine = halotools_wtpcf(xyz, r_bins, period=rL)

    plt.figure()
    plt.loglog(r_bins_cen, tpcf_ht, '-', lw=2)
    plt.loglog(r_bins_cen, wtpcf_mine, ':kx', lw=2)
    slct = x<50
    weights[~slct]=0

    tpcf_ht = tpcf(xyz[slct,:], r_bins, period=rL)
    tpcf_mine1 = halotools_tpcf(xyz[slct, :], r_bins, rL)#, weights=weights)
    wtpcf_mine2 = halotools_wtpcf(xyz, r_bins, rL, weights=weights)
    plt.figure()
    plt.loglog(r_bins_cen, tpcf_ht, '-', lw=2, label='Halotools tpcf')
    plt.loglog(r_bins_cen, tpcf_mine1, ':kx', lw=2, label='npair')
    plt.loglog(r_bins_cen, wtpcf_mine2, ':rx', lw=2, label='marked_npair')


    ht_wp = wp(xyz, r_bins, pi_max=15, period=rL)
    np_wp = halotools_wprp(xyz, r_bins, pi_max=15, period=rL)
    plt.figure()
    plt.loglog(r_bins_cen, ht_wp, '-', lw=2, label='Halotools wp')
    print(np.shape(r_bins_cen), np.shape(np_wp))
    print(np_wp)
    print(np_wp/ht_wp)
    plt.loglog(r_bins_cen, np_wp, ':kx', lw=2, label='npair wp')
    
    plt.show()
    print("done??")

    


    # cosmology.setCosmology('WM AP7')
    # core_xyz, core_m_infall, core_radius, core_hmass = get_cores()
    # slct = core_m_infall > 1e13
    # print("original size: ", len(slct))
    # print("reduced size: ", np.sum(slct))
    # core_xyz = core_xyz[slct, :]
    # core_m_infall = core_m_infall[slct]
    # core_radius = core_radius[slct]
    # slct = core_radius < 0.03
