import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import sys
import time

from astropy.cosmology import WMAP7

from astropy.cosmology import WMAP7

## All in i-band

def press_schecter(norm, Mstar, alpha):
    return lambda M: 0.4*np.log(10) * norm * 10**(0.4 * (Mstar -M)*(alpha+1.0)) * np.exp(-10**(0.4*(Mstar-M)))


def luminosity_function_blanton_funct():
    # https://arxiv.org/pdf/astro-ph/0210215.pdf
    #return press_schecter(0.0147, -20.82, -1.0) Uncorrected for h=1
    return press_schecter(0.00504/0.7**3, -20.04, -1.0) 

# def luminosity_function_montero_dorta():
#     return press_schecter(0.0109, -20.97, -1.16)
# https://arxiv.org/abs/0806.4930
#luminosity_function_montero_dorta = lambda x: press_schecter(0.0109, -20.97, -1.16)(x) #h=1.0
luminosity_function_montero_dorta = lambda x: press_schecter(0.0037387/0.7**3, -20.16, -1.16)(x-5*np.log10(0.7)) #h=0.7
luminosity_function_blanton =       lambda x: press_schecter(0.00504  /0.7**3, -20.04, -1.00)(x-5*np.log10(0.7))

# def luminosity_function_blanton(mag_i):
#     #return luminosity_function_blanton_funct()(mag_i) uncorrected for h=1
#     return luminosity_function_blanton_funct()(mag_i)

def convert_physical_to_comoving_vol( end_z):
    z_bins = np.linspace(0, end_z, 1000)
    comov_vol = WMAP7.comoving_volume(z_bins)
    comov_shell = comov_vol[:-1] - comov_vol[1:]
    z_shell = (z_bins[:-1] + z_bins[1:])/2.0
    a_shell = 1.0/(1.0+z_shell)
    a3_shell = a_shell**3
    total_comov_vol = np.sum(comov_shell)
    total_phys_vol = np.sum(comov_shell*a3_shell)
    return total_phys_vol/total_comov_vol

def get_expected_abundance(mag, author="Balton"):
    mags = np.linspace(-25, -15, 100000)
    dmags = mags[1]-mags[0]
    a = 1./(1.+0.21)
    factor = convert_physical_to_comoving_vol(0.21)
    if author=="Balton":
        lum_func = luminosity_function_blanton(mags)
    elif author=="Monter":
        lum_func = luminosity_function_montero_dorta(mags)
    else:
        raise KeyError
    cmltv_sum = np.cumsum(lum_func*dmags)
    interp_val = np.interp(mag, mags, cmltv_sum)
    return interp_val/factor
        

