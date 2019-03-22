

from __future__ import print_function, division
import numpy as np
from halotools.mock_observables import marked_tpcf, tpcf, marked_npairs_3d, npairs_3d, wp, marked_npairs_xy_z, npairs_xy_z

def halotools_wprp(core_xyz, rp_bins, pi_max, period=None):
    cnts = npairs_xy_z(core_xyz, core_xyz, rp_bins, np.array([0,pi_max]), period=period)
    cnts_shell = np.diff(cnts[:,1])
    r_bins_vol = np.pi*rp_bins**2
    r_bins_shell_vol = np.diff(r_bins_vol)
    Npts = np.shape(core_xyz)[0]
    expected_density = Npts/period**3
    expected_pair_cnts = Npts*r_bins_shell_vol*expected_density
    return cnts_shell/expected_pair_cnts - 1.0

def halotools_weighted_wprp(core_xyz, rp_bins, pi_max, period=None, weights=None):
    wcnts = marked_npairs_xy_z(core_xyz, core_xyz, rp_bins, np.array([0,pi_max]), period=period, weights=weights)
    wcnts_shell = np.diff(cnts)
    r_bins_vol = np.pi*rp_bins**2
    r_bins_shell_vol = np.diff(r_bins_vol)
    if weights is None:
        Npts = np.shape(xyz)[0]
    else:
        Npts = np.sum(weights)
    expected_density = Npts/period**3
    expected_pair_cnts = Npts*r_bins_shell_vol*expected_density
    return wcnts_shell/expected_pair_cnts - 1.0

def halotools_wtpcf(xyz, r_bins, period, weights=None):
    wcnts = marked_npairs_3d(xyz, xyz, r_bins, period=period, weights1=weights, weights2=weights, weight_func_id=1)
    wcnts_bins = wcnts[1:]-wcnts[:-1]
    r_bins_vol = 4.0/3.0*np.pi*r_bins**3
    r_bins_shell_vol = r_bins_vol[1:]-r_bins_vol[:-1]
    if weights is None:
        Npts = np.shape(xyz)[0]
    else:
        Npts = np.sum(weights)
    expected_density = Npts/period**3
    expected_pair_wcnts = Npts*r_bins_shell_vol*expected_density
    return wcnts_bins/expected_pair_wcnts - 1.0

def halotools_tpcf(xyz, r_bins, period):
    wcnts =npairs_3d(xyz, xyz, r_bins, period=period)
    wcnts_bins = wcnts[1:]-wcnts[:-1]
    r_bins_vol = 4.0/3.0*np.pi*r_bins**3
    r_bins_shell_vol = r_bins_vol[1:]-r_bins_vol[:-1]
    Npts = np.shape(xyz)[0]
    expected_density = Npts/period**3
    expected_pair_wcnts = Npts*r_bins_shell_vol*expected_density
    return wcnts_bins/expected_pair_wcnts - 1.0
