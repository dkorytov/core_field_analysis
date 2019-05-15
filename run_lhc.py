#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sys

from core_model import *

if __name__ == "__main__":
    # rp_bins = np.logspace(-1, 1, 16)
    outfile = sys.argv[1]
    n_sample = np.int(sys.argv[2])
    # lhc = create_lhc(n_sample)
    # run_and_save_lhc(outfile, lhc, rp_bins)
    lhc, rp_bins, wps = load_lhc_run(outfile)
    # gp_model = create_gp(lhc, wps)
    # save_gp(outfile.replace(".hdf5", ".gp.pckl"), gp_model)
    load_gp(outfile.replace(".hdf5", ".gp.pckl"))
