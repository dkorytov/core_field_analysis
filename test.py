#!/usr/bin/env python3

# from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from calc_wp import *
from corrfunc_interface import * 
from colossus.cosmology import cosmology


if __name__ == "__main__":
    print("hey")
    cosmology.setCosmology('WMAP7')
    core_xyz, core_m_infall, core_radius, core_hmass = get_cores()
    core_ra, core_dec, core_cz = get_core_ra_dec(core_xyz)
    r_bins = get_gao_r_bins()
    r_bins_cen = dtk.bins_avg(r_bins)
    corrfunc_wp = calc_corrfunc_wp(256.0/0.7, r_bins, core_ra, core_dec, core_cz)
