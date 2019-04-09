from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py

from pyDOE2 import lhs 
import GPy
from GPy.kern import RBF
from GPy.models import GPRegression
import pickle
import time


from abundance import get_expected_abundance
from halotools_tpcf_interface import *
from calc_wp import *


def sigmoid(x, x0, k, sign):
    y = 1./(1.+np.exp((k/x0)*(x-x0)*sign))
    return y

def get_core_model_hardtrans(core_dict, model_dict):
    weights = np.ones_like(core_dict['m_eff'], dtype='f4')
    slct = (core_dict['m_eff']>model_dict['m_cut']) & (core_dict['r_eff'] < model_dict['r_cut'])
    weights[slct] = 0
    return weights

def get_core_model_softtrans(core_dict, model_dict):
    w_mass = sigmoid(core_dict['m_eff'], model_dict['m_cut'], model_dict['m_cut_k'], -1)
    w_rad  = sigmoid(core_dict['r_eff'], model_dict['r_cut'], model_dict['r_cut_k'], +1)
    weight = w_mass * w_rad
    return weight

def get_core_hod_weight(mass_bins, hod_halo_cnt, core_hmass, weights):
    core_cnt, _ = np.histogram(core_hmass, bins=mass_bins, weights=weights)
    return core_cnt/hod_halo_cnt

def calc_abundance_distance_weights(mag, rL, weights):
    if mag not in calc_abundance_distance_weights._cache:
        expected_abund = get_expected_abundance(mag) 
        calc_abundance_distance_weights._cache[mag] = expected_abund
    else:
        expected_abund = calc_abundance_distance_weights._cache[mag] 
    vol = rL*rL*rL
    core_number = np.sum(weights)
    abund = core_number/vol
    return ((abund-expected_abund)/(0.1*expected_abund))**2

calc_abundance_distance_weights._cache = {}


def augment_lhc_axis(design, dim, min_v, max_v):
    design[:,dim] = design[:,dim]*(max_v-min_v) + min_v
    return design

def create_lhc(ntrain):
    design = lhs(4, ntrain, criterion='m')
    print(design)
    augment_lhc_axis(design, 0, 12.2, 12.5)
    augment_lhc_axis(design, 1, 1, 5)
    augment_lhc_axis(design, 2, 0.06, 0.5)
    augment_lhc_axis(design, 3, -1, 4)
    return design

def create_model_dict_from_lhc(lhc_slice):
    model_dict =  {}
    model_dict['m_cut'] = lhc_slice[0]
    model_dict['m_cut_k'] = 10**lhc_slice[1]
    model_dict['r_cut'] = lhc_slice[2]
    model_dict['r_cut_k'] = 10**lhc_slice[3]
    return model_dict


def run_lhc(lhc, rp_bins):
    n_sample = lhc.shape[0]
    core_xyz, core_m_infall, core_lgm_infall, core_radius, core_hmass = get_cores()
    core_dict= {"m_eff": core_lgm_infall, 
                "r_eff": core_radius}
    rp_bins_cen = dtk.bins_avg(rp_bins)
    wps = np.zeros((n_sample,len(rp_bins_cen)), dtype=np.float)
    t00 = time.time()
    for i in range(0, n_sample):
        print(i)
        t0 = time.time()
        model_dict = create_model_dict_from_lhc(lhc[i,:])
        weights = get_core_model_softtrans(core_dict, model_dict)
        core_wp = halotools_weighted_wprp(core_xyz, rp_bins, 20, period=256, weights=weights, fast_cut=0.001)
        wps[i,:] = core_wp
        t1 = time.time()
        print("\ttime: ", t1-t0)
        samples_left = i-n_sample-1
        time_per_sample = (t00-t1)/(i+1)
        print("\tETA: ", samples_left*time_per_sample)
    return wps

def run_and_save_lhc(outfile, lhc, rp_bins):
    wps = run_lhc(lhc, rp_bins)
    save_lhc_run(outfile, lhc, rp_bins, wps)

def save_lhc_run(outfile, lhc, rp_bins, wps):
    hfile = h5py.File(outfile, 'w')
    hfile['rp_bins'] = rp_bins
    hfile['lhc'] = lhc
    hfile['wps'] = wps
    hfile.close()

def load_lhc_run(infile):
    hfile = h5py.File(infile, 'r')
    rp_bins = hfile['rp_bins'].value
    lhc = hfile['lhc'].value
    wps  = hfile['wps'].value
    return lhc, rp_bins, wps

def create_gp(lhc, values):
    ndim = np.shape(lhc)[1]
    assert lhc.shape[0] == values.shape[0], "lhc and values must share the same length. Their shapes are lhc:{} and values:{}".format(lhc.shape,values.shape)
    kernel = RBF(input_dim=ndim)
    m = GPRegression(lhc, values, kernel)
    __=m.optimize_restarts(10, verbose=False)
    return m

def save_gp(outfile, gp_model):
    gp_model.pickle(outfile)

def load_gp(infile):
    return GPy.load(infile)
