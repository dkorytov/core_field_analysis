#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from scipy.optimize import minimize
import time

from colossus.cosmology import cosmology

from calc_wp import *
from halotools_tpcf_interface import *
from core_model import *

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

def test_wp():
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

def test_hod_model():
    core_xyz, core_m_infall, core_lgm_infall, core_radius, core_hmass = get_cores()
    mass_bins = np.logspace(12.1, 15, 64)
    mass_bins_cen = dtk.bins_avg(mass_bins)
    halo_tags, halo_mass = load_fof("/media/luna1/dkorytov/data/AlphaQ/fof/m000-499.fofproperties")
    hod_halo_cnt, _ = np.histogram(halo_mass, bins=mass_bins)

    core_dict= {"m_eff": core_lgm_infall, 
                "r_eff": core_radius}
   
    m_infall = 12.2
    r_disrupt = 0.03
    model_dict={"m_cut": m_infall, 
                "m_cut_k": 1000,
                "r_cut": r_disrupt,
                "r_cut_k": 1000}
    weights = get_core_model_softtrans(core_dict, model_dict)
    hod = get_core_hod(mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, 10**m_infall, r_disrupt)
    whod = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)
    plt.figure()
    print(hod)
    print(whod)
    plt.loglog(mass_bins_cen, hod, lw=4)
    m_k_range = np.logspace(3,0,16)
    print(m_k_range)
    for m_k in m_k_range:
        model_dict['m_cut_k']=m_k
        weights = get_core_model_softtrans(core_dict, model_dict)
        whod = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)
        plt.loglog(mass_bins_cen, whod, "-r", alpha=0.3)
    r_k_range = np.logspace(1, -3, 16)
    model_dict['m_cut_k']=1000
    for r_k in r_k_range:
        model_dict['r_cut_k']=r_k
        weights = get_core_model_softtrans(core_dict, model_dict)
        whod = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)
        plt.loglog(mass_bins_cen, whod, "-k", alpha=0.3)
    
    plt.show()

def test_lhc(n_sample = 300):
    def calc_wp_cost(rp_bins, core_xyz, weights, data_wp, return_wp=False):
        core_wp = halotools_weighted_wprp(core_xyz, rp_bins, 20, period=256, weights=weights, fast_cut=0.01)
        cost= calc_distance(data_wp, core_wp)
        if return_wp:
            return cost, core_wp
        else:
            return cost

    def plot_stuff(m_bins_cen, wps, labels, title=None):
        plt.figure()
        for wp, label in zip(wps, labels):
            plt.loglog(m_bins_cen, wp, label=label)
        plt.legend(loc='best')
        if title is not None:
            plt.title(title)


    # n_sample = 300
    # n_sample = lhc.shape[0]
    # lhc, rp_bins, wps = load_lhc_run("cache/lhc.wps.big_run.1.hdf5")
    # design = lhc
    rp_bins = np.logspace(-1,1,16)
    design = create_lhc(n_sample)

    gp_model = load_gp('cache/lhc.wps.big_run.1.gp.pckl')


    
    print(np.shape(design))
    core_xyz, core_m_infall, core_lgm_infall, core_radius, core_hmass = get_cores()
    mass_bins = np.logspace(12, 15, 32)
    mass_bins_cen = dtk.bins_avg(mass_bins)
    halo_tags, halo_mass = load_fof("/media/luna1/dkorytov/data/AlphaQ/fof/m000-499.fofproperties")
    hod_halo_cnt, _ = np.histogram(halo_mass, bins=mass_bins)
    
    data_hod = get_cac09_bp_hod(mass_bins, threshold=Ldot_from_mstar(0))

    core_dict= {"m_eff": core_lgm_infall, 
                "r_eff": core_radius}
    design_cost = np.zeros(n_sample)
    design_best = None
    design_best_cost = np.inf
    # rp_bins = np.logspace(-1,1.5, 32)
    rp_bins_cen = dtk.bins_avg(rp_bins)
    cac09_wp =  get_cac09_md_2pt(rp_bins, threshold=Ldot_from_mstar(0))
    
    print("starting latin hyper cube")
    for i in range(0, n_sample):
        print(i)
        t0 = time.time()
        model_dict = create_model_dict_from_lhc(design[i,:])
        weights = get_core_model_softtrans(core_dict, model_dict)
        t1 = time.time()
        core_whod = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)
        cost1 = calc_distance(data_hod, core_whod, max_dist=2.0)
        t2 = time.time()
        cost2 = calc_abundance_distance_weights(-21.22, 256.0, weights)
        t3 = time.time()
        # cost3, core_wp = calc_wp_cost(rp_bins, core_xyz, weights, cac09_wp, return_wp=True)
        # core_wp = wps[i]
        core_wp = gp_model.predict(design[i,:].reshape((1,4)))
        cost3 = calc_distance(cac09_wp, core_wp)
        # cost3 = 0
        t4 = time.time()
        cost = cost2 + cost3
        design_cost[i]= cost
        print(" ==   model   ==")
        print("\tm_eff  :  {:.2f}".format(design[i,0]))
        print("\tm_eff_k:  {:.2f}".format(design[i,1]))
        print("\tr_eff  :  {:.3f}".format(design[i,2]))
        print("\tr_eff_k:  {:.2f}".format(design[i,3]))
        print("==cost: {:.2f}==".format(cost))
        print("\t\tabund: {:.2f}".format(cost2))
        print("\t\thod  : {:.2f}".format(cost1))
        print("\t\twprp : {:.2f}".format(cost3))
        print("==time: {:.3f}==".format(t4-t0))
        print("\t\tweights: {:.3f}".format(t1-t0))
        print("\t\thod    : {:.3f}".format(t2-t1))
        print("\t\tabund  : {:.3f}".format(t3-t2))
        print("\t\twp(rp) : {:.3f}".format(t4-t3))
        print("\n")
        if cost < design_best_cost:
            design_best_cost = cost
            design_best = design[i,:]

            # plot_stuff(mass_bins_cen, [data_hod, core_whod], ['cac09', 'core'],title="HOD: {:.3f}/{:.3f}".format(cost1, cost))
            # plot_stuff(rp_bins_cen, [cac09_wp, core_wp], ['cac09', 'core'], title="WP: {:.3f}/{:.3f}".format(cost3, cost))

            # plt.show()

    print("Latin hyper cube winners:")
    print(np.min(design_cost), np.max(design_cost))
    print(np.log10(np.min(design_cost)), np.log10(np.max(design_cost)))
    # print("starting minimizer")
    # def func_to_minimize(args):
    #     model_dict = create_model_dict(args)
    #     weights = get_core_model_softtrans(core_dict, model_dict)
    #     core_whod = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)
    #     cost = calc_distance(data_hod, core_whod, max_dist=2.0)
    #     return cost
    # res = minimize(func_to_minimize, design_best, options={"maxiter": 1})

    # print("minimized result: ")
    # print(res)
    # x_best = res.x
    # model_dict = create_model_dict(x_best)
    # weights = get_core_model_softtrans(core_dict, model_dict)
    # core_whod_minimized = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)

    srt = np.argsort(-design_cost)
    plt.figure()
    plt.scatter(design[:,0][srt], design[:, 2][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    plt.xlabel('m_infall');plt.ylabel('r_disrupt')
    plt.colorbar()

    plt.figure()
    plt.scatter(design[:,0][srt], design[:, 1][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    plt.xlabel('m_infall');plt.ylabel('m_infall_k')
    plt.colorbar()

    plt.figure()
    plt.scatter(design[:,2][srt], design[:, 3][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    plt.xlabel('r_disrupt');plt.ylabel('r_disrupt_k')
    plt.colorbar()

    plt.figure()
    plt.scatter(design[:,1][srt], design[:, 3][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    plt.xlabel('m_infall_k');plt.ylabel('r_disrupt_k')
    plt.colorbar()

    fig, axs = plt.subplots(3,3, figsize=(10,10))
    axs[0,0].scatter(design[:,0][srt], design[:, 1][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    axs[1,0].scatter(design[:,0][srt], design[:, 2][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    axs[2,0].scatter(design[:,0][srt], design[:, 3][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    axs[0,1].set_visible(False)
    axs[1,1].scatter(design[:,1][srt], design[:, 2][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    axs[2,1].scatter(design[:,1][srt], design[:, 3][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    axs[0,2].set_visible(False)
    axs[1,2].set_visible(False)
    axs[2,2].scatter(design[:,2][srt], design[:, 3][srt], c=design_cost[srt], cmap='nipy_spectral_r', s=500, edgecolors='none', norm=clr.LogNorm())
    for i in range(0,3):
        for j in range(0,3):
            ax = axs[i,j]
            if j != 0:
                ax.set_yticklabels([])
            if i != 2:
                ax.set_xticklabels([])
    axs[0,0].set_ylabel('m_k')
    axs[1,0].set_ylabel('r_cut')
    axs[2,0].set_ylabel('r_k')
    axs[2,0].set_xlabel('m_infall')
    axs[2,1].set_xlabel('m_k')
    axs[2,2].set_xlabel('r_cut')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)


    weights_best = get_core_model_softtrans(core_dict, create_model_dict_from_lhc(design_best))
    # plt.figure()
    # print(design_best)
    # print(design_best_cost)

    core_whod_best = get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights)
    # plt.title(str(design_best_cost))
    # plt.loglog(mass_bins_cen, data_hod, label='data')
    # plt.loglog(mass_bins_cen, core_whod_best, label='Best Latin Cube')
    # # plt.loglog(mass_bins_cen, core_whod_minimized, label='Minimized')
    # plt.legend(loc='best')

    fig = plt.figure()
    grid = plt.GridSpec(4,1, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(grid[:3,:])
    ax.loglog(mass_bins_cen, data_hod, label='data')
    ax.loglog(mass_bins_cen, core_whod_best, label='Best Latin Cube')
    ax.legend(loc='best')
    ax.set_ylabel("HOD")
    ax_diff = fig.add_subplot(grid[3,:])
    ax_diff.semilogx(mass_bins_cen, np.log10(core_whod_best/data_hod))
    ax_diff.axhline(1, ls='--', color='k')
    ax_diff.set_xlabel("M180 [Msun/h, h=1]")
    ax_diff.set_ylabel("log10(core fit / data)")

    core_wwp = halotools_weighted_wprp(core_xyz, rp_bins, 20, period=256, weights=weights_best)
    # plt.figure()
    # plt.loglog(rp_bins_cen, cac09_wp, label='cac09')
    # print("ht_wwprp input: ", np.shape(core_xyz), np.shape(rp_bins))
    # print("output: ", np.shape(core_wwp), weights)
    # print(core_wwp)
    # plt.loglog(rp_bins_cen, core_wwp, label='Best Latin Cube')
    # plt.legend(loc='best')
    
    fig = plt.figure()

    ax = fig.add_subplot(grid[:3,:])
    ax.loglog(rp_bins_cen, cac09_wp, label='cac09')
    ax.loglog(rp_bins_cen, core_wwp, label='Best Latin Cube')
    ax.legend(loc='best')
    ax.set_ylabel('wp(rp)')
    ax_diff = fig.add_subplot(grid[3,:], sharex = ax)
    ax_diff.semilogx(rp_bins_cen, (core_wwp-cac09_wp)/cac09_wp)
    ax_diff.set_ylabel("(fit - data)/data")
    ax_diff.set_xlabel("rp [Mpc/h, h=1]")
    ax_diff.axhline(0, ls='--', color='k')
    plt.show()
    


if __name__ == "__main__":
    # test_hod_model()
    # test_lhc(n_sample=10)
    
    # run_and_save_lhc("cache/wps.test2.hdf5", 400, np.logspace(-1, 1, 16))a
    # exit()
    test_lhc(n_sample = 300)
    exit()
    lhc, rp_bins, wps = load_lhc_run("cache/wps.test2.hdf5")
    rp_bins_cen = dtk.bins_avg(rp_bins)
    # gp_model = create_gp(lhc, wps)
    # save_gp("cache/gp_model.pckl", gp_model)
    gp_model = load_gp("cache/lhc.wps.big_run.1.gp.pckl")
    # plt.figure()
    # for i in range(0, wps.shape[0]):
    #     plt.loglog(rp_bins_cen, wps[i,:], alpha=0.3)
    # plt.show()
    lhc2, _, wps2 = load_lhc_run("cache/wps.test.hdf5")
    
    for i in range(0, wps2.shape[0]):
        f,axs = plt.subplots(2,1)

        wps_gp = gp_model.predict(lhc2[i].reshape((1,4)))[0][0]

        axs[0].loglog(rp_bins_cen, wps2[i], label="True Value")
        axs[0].loglog(rp_bins_cen, wps_gp, label="Emulated values")
        axs[1].semilogx(rp_bins_cen, wps_gp/wps2[i])
        axs[1].axhline(1.0, ls='--', color='k')
        plt.show()
    # cosmology.setCosmology('WM AP7')
    # core_xyz, core_m_infall, core_radius, core_hmass = get_cores()
    # slct = core_m_infall > 1e13
    # print("original size: ", len(slct))
    # print("reduced size: ", np.sum(slct))
    # core_xyz = core_xyz[slct, :]
    # core_m_infall = core_m_infall[slct]
    # core_radius = core_radius[slct]
    # slct = core_radius < 0.03
