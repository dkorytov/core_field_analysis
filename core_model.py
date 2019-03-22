from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py


def sigmoid(x, x0, k):
    y = 1./(1.+np.exp(k*(x-x0)))
    return y

def get_core_model_hardtrans(core_dict, model_dict):
    weights = np.ones_like(core_dict['m_eff'], dtype='f4')
    slct = (core_dict['m_eff']>model_dict['m_cut']) & (core_dict['r_eff'] < model_dict['r_cut'])
    weights[slct] = 0
    return weights

def get_core_model_softtrans(core_dict, model_dict):
    w_mass = sigmoid(core_dict['m_eff'], model_dict['m_cut'], model_dict['m_cut_k'])
    w_rad  = sigmoid(core_dict['r_eff'], model_dict['r_cut'], model_dict['r_cut_k'])
    weight = w_mass * w_rad
    return weight
