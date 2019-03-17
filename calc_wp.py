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

from abundance import get_expected_abundance

r_bin_edges = np.logspace(-1, 1.5, 30)
r_bin_cent = dtk.log_bins_avg(r_bin_edges)


def calc_abundance_distance(mag, rL, core_m_infall, core_radius, m_infall, r_disrupt):
    """ Units are in Mpc/h, h=0.7
    """
    if mag not in calc_abundance_distance._cache:
        expected_abund = get_expected_abundance(mag)
        calc_abundance_distance._cache[mag] = expected_abund
    else:
        expected_abund = calc_abundance_distance._cache[mag] 
    vol = rL*rL*rL
    core_num=np.sum((core_m_infall > m_infall) & (core_radius < r_disrupt))
    abund = core_num/vol
    return ((expected_abund-abund)/(0.1 * expected_abund))**2

calc_abundance_distance._cache = {}
def Ldot_from_mstar(offset=0):
    return 10.352+offset/-2.5

def calc_2pt(r_disrupt = .035, m_infall_cut = 10**12.2, fig = None,):
    if fig is None:
        fig = plt.figure()
    print("we will be calculating the two point funciton")
    # core_loc = "/media/luna1/dkorytov/data/OuterRim/cores_500L/03_31_2018.OR.401.coreproperties"
    core_loc = "/media/luna1/rangel/new_cores_AlphaQ/07_13_17.AlphaQ.401.coreproperties"
    x = dtk.gio_read(core_loc, 'x')
    y = dtk.gio_read(core_loc, 'y')
    z = dtk.gio_read(core_loc, 'z')
    radius = dtk.gio_read(core_loc, 'radius')
    m_infall = dtk.gio_read(core_loc, 'infall_mass')
    rL = 256/0.70
    slct = (radius < r_disrupt) & (m_infall > m_infall_cut)
    slct = slct & (x>0) & (y>0) & (z>0)
    x = x / 0.70
    y = y / 0.70
    z = z / 0.70
    core_xyz = return_xyz_formatted_array(x[slct], 
                                             y[slct],
                                             z[slct],)
    print("Our 2pt results")
    # xi = tpcf(core_xyz, r_bin_edges, period=rL)
    xi = wp(core_xyz, r_bin_edges, pi_max=50, period=rL)
    if fig is None:
        fig = plt.figure()
    plt.plot(r_bin_cent, xi, label=r'${{\rm Cores fit: M_{{infall}}>10^{{{:.2f}}}, R_{{radius}}<{:.2f} }}$'.format(np.log10(m_infall_cut), r_disrupt))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel(r'${\rm r\ [Mpc/h]}$')
    # plt.ylabel(r'${\xi(r)}$')
    plt.ylabel(r'${\omega_p(r)}$')
    return fig

def calc_2pt_md(threshold=10.75, fig = None):
    print('starting here')
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='multidark', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    print('starting 2pt')
    # xi = tpcf(pos, r_bin_edges, period=halocat.Lbox)
    xi = wp(pos, r_bin_edges, pi_max=50, period=halocat.Lbox)
    if(fig is None):
        fig = plt.figure()
    plt.plot(r_bin_cent, xi, label=r'${{ \rm Cacciato+09:\ L > 10^{{{:.2f}}}L_{{\odot}} }}$'.format(threshold))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel(r'${\rm r\ [Mpc/h]}$')
    # plt.ylabel(r'${\xi(r)}$')
    plt.ylabel(r'${\omega_p(r)}$')
    return fig
   
def calc_1d_md(threshold=10.75, fig = None):
    print("calculating 1pt stats")
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='multidark', redshift=0.0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    def min_max(data):
        return np.min(data), np.max(data)
    print("x:", min_max(model.mock.galaxy_table['x']))
    print("y:", min_max(model.mock.galaxy_table['y']))
    print("z:", min_max(model.mock.galaxy_table['z']))
    print("count: ", len(model.mock.galaxy_table['x']))
    print("volume: ", 1e3**3)
    print("Density, h=1:", len(model.mock.galaxy_table['x'])/(1e3**3))
    print("Density, h=0.7:",len(model.mock.galaxy_table['x'])/(1e3**3)/0.7**3)
    print("Density: number / comoving volume")

def download_mock():
    print('downloading halo cat')
    from halotools.sim_manager import DownloadManager
    dman = DownloadManager()
    dman.download_processed_halo_table('multidark', 'rockstar', 0.0)
    print('done downloading')

def calc_distance(y1, y2, max_dist = None):
    if max_dist is None:
        return np.nansum((np.log10(y1)-np.log10(y2))**2)
    else:
        dists = (np.log10(y1)-np.log10(y2))**2
        dists[dists>max_dist]=max_dist
        return np.nansum(dists)

def combine_error(y1_err, y2_err):
    if y1_err is None:
        return y2_err
    elif y2_err is None:
        return y1_err
    else:
        return np.sqrt(y1_err**2 + y2_err**2)
        
def calc_distance_err(y1, y2, y1_err=None,  y2_err=None):
    if y1_err is None and y2_err is None:
        return calc_distance(y1, y2)
    err_tot = combine_err(y1_err, y2_err)
    return np.sum(((y1-y2)/err_tot)**2)

def get_cac09_md_2pt(r_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='multidark', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    print('starting 2pt')
    # xi = tpcf(pos, r_bin_edges, period=halocat.Lbox)
    xi = wp(pos, r_bins, pi_max=50, period=halocat.Lbox)
    return xi

def get_cac09_bp_2pt(r_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='bolplanck', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    print('starting 2pt')
    # xi = tpcf(pos, r_bin_edges, period=halocat.Lbox)
    xi = wp(pos, r_bins, pi_max=50, period=halocat.Lbox)
    print("wp(r): ", xi)
    plt.figure()
    plt.plot(r_bins[:-1], xi)
    return xi

def get_gao_2pt():
    """ from https://arxiv.org/pdf/1401.3009.pdf """
    wp = np.array([10029.85, 4744.7, 2860.37, 2732.31, 1560.89, 1025.62, 629.37, 363.88, 195.87, 128.8, 92.67, 67.65, 48.11, 32.13, 19.56, 10.59, 3.73, 1.24])
    wp_err = np.array([7420.09, 883.75, 359.76, 236.19, 105.03, 60.78, 30.65, 12.43, 6.92, 4.58, 2.99, 2.09, 1.55, 1.33, 1.13, 0.87, 0.62, 0.5])
    return wp, wp_err

def get_gao_r_bins():
    r_bins = np.logspace(-1.77, 1.8, 19)
    # print(dtk.bins_avg(r_bins))
    return r_bins

def get_core_2pt(r_bins,  core_xyz, core_m_infall, core_radius, core_hmass, m_infall, r_disrupt):
    slct = (core_m_infall > m_infall) & (core_radius < r_disrupt)
    core_xyz = core_xyz[slct, :]
    if(core_xyz.size == 0):
        xi =  np.empty((len(r_bins)-1))
        xi[:] = np.nan
    else:
        xi = wp(core_xyz, r_bins, pi_max=50, period=256/0.7, num_threads='max')
    return xi

def get_core_hod(mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, m_infall, r_disrupt):
    slct = (core_m_infall > m_infall) & (core_radius < r_disrupt)
    core_cnt, _= np.histogram(core_hmass[slct], bins=mass_bins)
    return core_cnt/hod_halo_cnt

def get_cac09_md_hod(mass_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='multidark', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    h_gal,  _  = np.histogram(model.mock.galaxy_table['halo_mvir'], bins=mass_bins)
    h_halo, _  = np.histogram(halocat.halo_table['halo_mvir'],      bins=mass_bins)
    return h_gal/h_halo

def get_cac09_bp_hod(mass_bins, threshold=None):
    model = PrebuiltHodModelFactory('cacciato09', threshold=threshold)
    halocat = CachedHaloCatalog(simname='bolplanck', redshift=0)
    halocat.halo_table['halo_m180b']=halocat.halo_table['halo_mvir']
    print(halocat.halo_table.keys())
    model.populate_mock(halocat)
    pos = return_xyz_formatted_array(model.mock.galaxy_table['x'], 
                                     model.mock.galaxy_table['y'], 
                                     model.mock.galaxy_table['z'], )
    h_gal,  _  = np.histogram(model.mock.galaxy_table['halo_mvir'], bins=mass_bins)
    h_halo, _  = np.histogram(halocat.halo_table['halo_mvir'],      bins=mass_bins)
    h = h_gal/h_halo
    print("HOD: ", h)
    print("h_gal:", h_gal)
    print("h_halo:", h_halo)
    plt.figure()
    plt.plot(mass_bins[:-1], h, )
    return h_gal/h_halo

def load_fof(fof_fname):
    fof_htag = dtk.gio_read(fof_fname, "fof_halo_tag")
    fof_mass = dtk.gio_read(fof_fname, "fof_halo_mass")
    m180b_mass, _, _ = mass_adv.changeMassDefinitionCModel(fof_mass/0.7*0.9, 0, 
                                                     "200c", "180m", 
                                                     c_model='child18')
    
    # df = pd.DataFrame({"fof_halo_tag": fof_htag,
    #                    "fof_halo_mass": fof_mass,
    #                    "m180b": m180b_mass})
    return fof_htag, m180b_mass

def get_cores():
    core_loc = "/media/luna1/rangel/new_cores_AlphaQ/07_13_17.AlphaQ.499.coreproperties"
    x = dtk.gio_read(core_loc, 'x')
    y = dtk.gio_read(core_loc, 'y')
    z = dtk.gio_read(core_loc, 'z')
    radius = dtk.gio_read(core_loc, 'radius')
    m_infall = dtk.gio_read(core_loc, 'infall_mass')
    htag = dtk.gio_read(core_loc, 'fof_halo_tag')
    x = x / 0.70
    y = y / 0.70
    z = z / 0.70
    core_xyz = return_xyz_formatted_array(x,y,z)
    fof_fname = "/media/luna1/dkorytov/data/AlphaQ/fof/m000-499.fofproperties"
    halo_tags, halo_mass = load_fof(fof_fname)
    srt = np.argsort(halo_tags)
    indx = dtk.search_sorted(halo_tags, htag, sorter=srt)
    print(np.sum(indx==-1), "number not found")
    return core_xyz, m_infall, radius, halo_mass[indx]

def get_cached_hod(fname, mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, m_infall, r_disrupt, force=False, write=True):
    hfile = h5py.File(fname, 'a')
    key="hod mi:{},rd:{}".format(m_infall, r_disrupt)
    if key in hfile and not force:
        core_hod= hfile[key].value
    else:
        core_hod = get_core_hod(mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, m_infall, r_disrupt)
        if write:
            hfile[key]=core_hod
    hfile.close()
    return core_hod

def get_cached_2pt(fname,  r_bins, core_xyz, core_m_infall, core_radius, m_infall, r_disrupt, force=False, write=True):
    hfile = h5py.File(fname, 'a')
    key="wp mi:{},rd:{}".format(m_infall, r_disrupt)
    if key in hfile and not force:
        core_wp= hfile[key].value
    else:
        core_wp = get_core_2pt(r_bins, core_xyz, core_m_infall, core_radius, m_infall, r_disrupt)
        if write:
            hfile[key]=core_wp
    hfile.close()
    return core_wp

def grid_scan(use_hod=False, use_wp=True, use_abundance=True, force=False):
    m_infall_bins_edges = np.logspace(11.0, 13.0, 12)
    r_disrupt_bins_edges = np.linspace(0.00, 0.1, 12)
    m_infall_bins = dtk.bins_avg(m_infall_bins_edges)
    r_disrupt_bins = dtk.bins_avg(r_disrupt_bins_edges)
    result = np.zeros((m_infall_bins.size, r_disrupt_bins.size))
    # r_bins = np.logspace(-0.5,1.2, 32)
    r_bins = get_gao_r_bins()
    r_bins_cen = dtk.bins_avg(r_bins)
    mass_bins = np.logspace(13, 15, 16)
    mass_bins_cen = dtk.bins_avg(mass_bins)
    core_xyz, core_m_infall, core_radius, core_hmass = get_cores()
    fof_fname = "/media/luna1/dkorytov/data/AlphaQ/fof/m000-499.fofproperties"
    halo_tags, halo_mass = load_fof(fof_fname)
    hod_halo_cnt, _ = np.histogram(halo_mass, bins=mass_bins)
    data_xi, data_xi_err = get_gao_2pt()
    data_xi_label= "Gao+XX"
    data_hod = get_cac09_bp_hod(mass_bins, threshold=Ldot_from_mstar(0))
    best_m_infall = 0
    best_r_disrupt = 0
    min_cost = 100
    for m_i, m_infall_v in enumerate(m_infall_bins):
        print("progress: ", m_i/len(m_infall_bins))
        for r_i, r_disrupt_v in enumerate(r_disrupt_bins):
            costs = []
            if use_wp:
                core_xi = get_cached_2pt("cache/gao_rbins.hdf5", r_bins, core_xyz, core_m_infall, core_radius,  m_infall_v, r_disrupt_v,force=force)
                costs.append( calc_distance(data_xi, core_xi))
            if use_hod:
                core_hod = get_cached_hod("cache/model1.hdf5", mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, m_infall_v, r_disrupt_v, force=force)
                costs.append(calc_distance(data_hod, core_hod, max_dist=2.0))
            if use_abundance:
                cost_abund = calc_abundance_distance(-21.6, 256, core_m_infall, core_radius, m_infall_v, r_disrupt_v)
                costs.append(cost_abund)
            cost = np.sum(costs)
            result[m_i, r_i] = cost
            print("\t", cost)
            if cost< min_cost:
                best_m_infall = m_infall_v
                best_r_disrupt = r_disrupt_v
                min_cost = cost
            # plt.figure()
            # plt.title("{:.2f}".format(cost))
            # plt.loglog(r_bins_cen, core_xi, label='core Mi:{:.2e} Rd:{:.2f}'.format(m_infall_v, r_disrupt_v))
            # plt.loglog(r_bins_cen, data_xi, label='data')
            # plt.legend()
            # plt.ylabel('wp(r)')
            # plt.xlabel('r [Mpc/h, comov, h=1]')

            # plt.figure()
            # plt.title("{:.2f}".format(cost))
            # plt.loglog(mass_bins_cen, core_hod, label='core Mi:{:.2e} Rd:{:.2f}'.format(m_infall_v, r_disrupt_v))
            # plt.loglog(mass_bins_cen, data_hod, label='data')
            # plt.legend()
            # plt.ylabel('HOD')
            # plt.xlabel('M180b [Msun]')
            # plt.show()


    abund_line_file = "../core_fit/tmp_hdf5/abundance=0.0024.hdf5"
    print(abund_line_file)
    hfile = h5py.File(abund_line_file, 'r')
    abund_infall_mass = hfile['abund_infall_mass'].value
    abund_radius      = hfile['abund_radius'].value
    profile_m_infall = 10**12.22
    profile_r_disrupt = 0.03
    result[~np.isfinite(result)]=np.nanmax(result)
    print(result)
    print("minfall: ", best_m_infall)
    print("best_r_disrupt:",  best_r_disrupt)
    print("cost: ", min_cost)

    plt.figure()
    best_xi = get_core_2pt(r_bins, core_xyz, core_m_infall, core_radius, best_m_infall, best_r_disrupt)
    plt.loglog(r_bins_cen, best_xi, label='core best fit')
    prof_xi = get_core_2pt(r_bins, core_xyz, core_m_infall, core_radius, profile_m_infall, profile_r_disrupt)
    plt.loglog(r_bins_cen, prof_xi, color='r', label='core profile fit')

    plt.loglog(r_bins_cen, data_xi, 'g', label=data_xi_label)
    plt.fill_between(r_bins_cen, data_xi+data_xi_err, data_xi-data_xi_err, alpha=0.3, color='g')
    plt.ylabel('wp(r)')
    plt.xlabel('r [Mpc/h, comov, h=1]')
    plt.legend(loc='best', framealpha=0.3)

    plt.figure()
    best_hod = get_core_hod(mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, best_m_infall, best_r_disrupt)
    plt.loglog(mass_bins_cen, best_hod, label='core best fit')
    prof_hod = get_core_hod(mass_bins, hod_halo_cnt, core_m_infall, core_radius, core_hmass, profile_m_infall, profile_r_disrupt)
    plt.loglog(mass_bins_cen, prof_hod, color='r', label='core profile fit')
    plt.loglog(mass_bins_cen, data_hod, label='data')
    plt.ylabel('HOD')
    plt.xlabel('M180b [Msun]')
    plt.legend(loc='best', framealpha=0.3)

    plt.figure()
    plt.title("Wp(r)+HOD cost, Min mass: {:.2e}".format(np.min(mass_bins)))
    plt.pcolor(m_infall_bins_edges, r_disrupt_bins_edges, result.T, cmap='nipy_spectral_r', norm=clr.LogNorm())
    plt.xscale('log')
    plt.colorbar()
    xlim = plt.xlim()
    ylim = plt.ylim()
    print(xlim)
    print(ylim)
    plt.plot(abund_infall_mass, abund_radius, '--k', lw=2, label='Abundance Req')
    plt.scatter([profile_m_infall], [profile_r_disrupt], s=[200], marker="*", edgecolors='k',c='r', label='profile fit' )
    plt.scatter([best_m_infall], [best_r_disrupt], s=[200], marker="*", edgecolors='k',c='b', label='current fit' )
    plt.legend(loc='best', framealpha=0.5)
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.ylabel('R_disrupt')
    plt.xlabel("M_infall")
    
    
    dtk.save_figs("figs/")
    plt.show()


if __name__ == "__main__":
    # from halotools.sim_manager import DownloadManager
    # dman = DownloadManager()
    # dman.download_processed_halo_table('bolplanck', 'rockstar', 0.0) 
    cosmology.setCosmology('WMAP7')
    grid_scan(use_wp=True)
    exit()
    Ldot_cut = np.log10(2.25e10)
    print(Ldot_cut)
    # calc_1d_md(Ldot_cut)
    

    fig = calc_2pt(m_infall_cut=10**12.22, r_disrupt=0.03 )
    calc_2pt_md(Ldot_from_mstar(0), fig=fig)
    plt.title("Mstar+0.0")

    fig =  calc_2pt( m_infall_cut=10**11.9, r_disrupt=0.03)
    calc_2pt_md(Ldot_from_mstar(0.5), fig=fig)
    plt.title("Mstar+0.5")

    fig =     calc_2pt(m_infall_cut=10**11.59, r_disrupt=0.03)
    calc_2pt_md(Ldot_from_mstar(1.0), fig=fig)

    plt.title("Mstar+1.0")


    # calc_2pt(fig, r_disrupt=0.01)
    # calc_2pt(fig, r_disrupt=0.02)
    # calc_2pt(fig, r_disrupt=0.03)
    # calc_2pt(fig, r_disrupt=0.04)
    # calc_2pt(fig, r_disrupt=0.05)
    # calc_2pt(fig, r_disrupt=0.06)
    # calc_2pt(fig, r_disrupt=0.07)


    # fig = calc_2pt_md(10.75)
    # calc_2pt(fig, m_infall_cut = 10**12.0)
    # calc_2pt(fig, m_infall_cut = 10**12.1)
    # calc_2pt(fig, m_infall_cut = 10**12.2)
    # calc_2pt(fig, m_infall_cut = 10**12.3)
    # calc_2pt(fig, m_infall_cut = 10**12.4)
    # calc_2pt(fig, m_infall_cut = 10**12.5)
    # calc_2pt(fig, m_infall_cut = 10**12.7)
    # calc_2pt(fig, m_infall_cut = 10**12.8)


    dtk.save_figs("figs/")
    plt.show()
