#!/usr/bin/env python2.7

from __future__ import print_function, division

import matplotlib
import os
#checks if there is a display to use.
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec
import dtk
import h5py
import time
import halotools

from plot_cmass_hod_sham import convert_step401_fofmass_to_m180b
from generate_hod_models import populate_halos_with_galaxies_with_NFW
from cmass_hod_sham import compute_cmass_hod_2pt
class MergerTree:
    def __init__(self, fname=None):
        if fname is not None:
            self.load_hdf5(fname)

    def load_hdf5(self, fname):
        eta = dtk.ETA()
        print("loading hdf5")
        hfile = h5py.File(fname,'r')
        hfh = hfile['forestHalos']
        self.desc_nodeIndex = hfh['descendentIndex'].value
        self.htag       = hfh['haloTag'].value
        self.pos        = hfh['position'].value
        self.step       = hfh['timestep'].value
        self.mass       = hfh['nodeMass'].value
        self.nodeIndex  = hfh['nodeIndex'].value
        # self.peak_mass  = hfh['nodeMass'].value
        self.progen     = [[] for x in range(len(self.mass))]
        eta.print_done()
        self.find_progenitors()

    def find_progenitors(self):
        eta = dtk.ETA()
        print("arg sorting")
        srt = np.argsort(self.nodeIndex)
        srt_desc = np.argsort(self.desc_nodeIndex)
        eta.print_done().reset()
        print("search sorted")
        desc_indexes = dtk.search_sorted(self.nodeIndex, self.desc_nodeIndex[srt_desc], sorter=srt)
        desc_indexes = desc_indexes[dtk.invert_sort(srt_desc)]
        # desc_indexes2 = dtk.search_sorted(self.nodeIndex, self.desc_nodeIndex, sorter=srt)

        # print(desc_indexes-desc_indexes2)
        # exit()
        eta.print_done().reset()
        # print(len(self.progen))
        for i in range(len(self.htag)):
            descn_index = desc_indexes[i]
            if descn_index != -1:
                self.progen[descn_index].append(i)
                assert descn_index != i, "hmm you are your own descn, I think not!"
        for i in range(len(self.htag)):
            if len(self.progen[i])>1:
                # self.print_progen(i)
                progen_masses = self.mass[self.progen[i]]
                most_massive_progen_index = np.argmax(progen_masses)
                a = self.progen[i][0]
                b = self.progen[i][most_massive_progen_index]
                self.progen[i][0]=b
                self.progen[i][most_massive_progen_index] = a

    def count_halo_infalls(self, i, mass_limit):
        # print(i)
        # print(mass_limit)
        # print(self.mass)
        # print(self.mass[i])
        if self.mass[i]<mass_limit:
            return 0
        progens = self.progen[i]
        tot_sum = 1 # We are counting ourselves. 
        for i, progen in enumerate(progens):
            assert progen != i, "hmm progen is itself?!"
            cnt = self.count_halo_infalls(progen, mass_limit)
            if i == 0: # The central doesn't get counted multiple times
                cnt = max(0, cnt-1)
            tot_sum += cnt
        return tot_sum

    def print_halo(self, i, prefix=""):
        print("{}Mass: {:.2e}, step: {}".format(prefix, self.mass[i], self.step[i]))

    def print_progen(self, i):
        print("Progens: ", len(self.progen[i]))
        for i, progen in enumerate(self.progen[i]):
            print("\tpregen:", i)
            self.print_halo(progen, prefix="\t\t")
                                               
    def get_halo_mass_and_infall_counts(self, mass_limit, step,
                                        last_isolated=False, return_position=False):
        step_slct = (self.step == step) & (self.mass > mass_limit)
        halo_indexes = np.where(step_slct)[0]
        total_count = 0
        masses = []
        counts  = []
        pos_x = []
        pos_y = []
        pos_z = []
        for i, halo_i in enumerate(halo_indexes):
            cnt = self.count_halo_infalls(halo_i, mass_limit)
            masses.append(self.mass[halo_i])
            counts.append(cnt)
            if return_position:
                pos_x.append(self.pos[halo_i, 0])
                pos_y.append(self.pos[halo_i, 1])
                pos_z.append(self.pos[halo_i, 2])

        masses = np.array(masses)
        counts = np.array(counts)
        if return_position:
            pos_x = np.array(pos_x)
            pos_y = np.array(pos_y)
            pos_z = np.array(pos_z)
            return masses, counts, pos_x, pos_y, pos_z
        else:
            return masses, counts
        
    def get_volume(self, step):
        slct_step = self.step == step
        print(np.shape(self.pos))
        x0,x1 = min_max(self.pos[slct_step, 0])
        y0,y1 = min_max(self.pos[slct_step, 1])
        z0,z1 = min_max(self.pos[slct_step, 2])
        dx = x1-x0
        dy = y1-y0
        dz = z1-z0
        volume = dx*dx*dy
        print("{:.2f}x{:.2f}x{:.2f}".format(dx,dy,dz))
        print("\t{:.2f}".format(volume))
        return volume
        

class MergerTreeGroup:
    def __init__(self, fname_pattern, num=None):
        self.merger_trees = []
        if num is None:
            self.merger_trees.append(MergerTree(fname_pattern))
        else:
            eta =dtk.ETA()
            for i in range(0, num):
                eta.print_progress(i,num)
                fname = fname_pattern.replace('${num}', str(i))
                print(fname)
                mt = MergerTree(fname)
                self.merger_trees.append(mt)
            eta.print_done()
            
    def get_halo_mass_and_infall_counts(self, infall, step, return_position=False):
        masses = []
        counts = []
        pos_x = []
        pos_y = []
        pos_z = []
        eta = dtk.ETA()
        mt_size = len(self.merger_trees)
        for i in range(0, mt_size):
            eta.print_progress(i, mt_size)
            if return_position:
                m, c, x, y, z = self.merger_trees[i].get_halo_mass_and_infall_counts(infall, step, return_position=True)
                pos_x.append(x)
                pos_y.append(y)
                pos_z.append(z)
            else:
                m,c = self.merger_trees[i].get_halo_mass_and_infall_counts(infall, step)
            masses.append(m)
            counts.append(c)
        eta.print_done()

        masses = np.concatenate(masses)
        counts = np.concatenate(counts)
        if return_position:
            pos_x = np.concatenate(pos_x)
            pos_y = np.concatenate(pos_y)
            pos_z = np.concatenate(pos_z)
            return masses, counts, pos_x, pos_y, pos_z
        else:
            return masses, counts

    def get_count_by_infall(self, infall, step):
        m,c = self.get_halo_mass_and_infall_counts(infall, step)
        return np.sum(c)

    def get_infall_by_count(self, target_count, step, min_mass=1e10,
                            max_mass=1e16, tolerance=0.01, min_count = None, max_count=None):
        """Binary search to match by expected count"""
        print("{:.2e} -> {:.2e}".format(min_mass, max_mass))
        if min_count is None:
            min_count = self.get_count_by_infall(min_mass, step)
        if max_count is None:
            max_count = self.get_count_by_infall(max_mass, step)
        mid_mass = dtk.log_avg([min_mass, max_mass])
        mid_count = self.get_count_by_infall(mid_mass, step)
        # print("\t ", target_count, mid_count)
        # print("\t diff: ", np.abs((mid_count-target_count)/target_count))

        if dtk.within_relative_tolerance(mid_count, target_count, tolerance):
            return mid_mass
        else:
            if mid_count>target_count:
                # search mid->max
                return self.get_infall_by_count(target_count, step, min_mass=mid_mass, min_count=mid_count, max_mass=max_mass, max_count=max_count)
            else:
                # search min->mid
                return self.get_infall_by_count(target_count, step, min_mass=min_mass, min_count=min_count, max_mass=mid_mass, max_count=mid_count)
        
    def get_volume(self, step):
        volume_tot = 0.0
        for mt in self.merger_trees:
            volume_tot += mt.get_volume(step)
        return volume_tot
    

def min_max(x):
    return np.min(x), np.max(x)

def concatenate_merger_trees(mt_list):
    mt_tot = MergerTree()
    tot_size = 0
    for mt in mt_list:
        tot_size += mt.htag.size
        
    mt_tot.descn_nodeIndex = np.zeros(tot_size, dtype=mt_list[0].descn_nodeIndex.dtype)
    mt_tot.htag      = np.zeros(tot_size, dtype=mt_list[0].htag.dtype)
    mt_tot.pos       = np.zeros(tot_size, dtype=mt_list[0].pos.dtype)
    mt_tot.step      = np.zeros(tot_size, dtype=mt_list[0].step.dtype)
    mt_tot.mass      = np.zeros(tot_size, dtype=mt_list[0].mass.dtype)
    mt_tot.nodeIndex = np.zeros(tot_size, dtype=mt_list[0].nodeIndex.dtype)
    mt_tot.progen    = [[] for x in range(tot_size)]
    
    offset = 0




def load_merger_trees(fname, num=None, vol_per_num=None,
                      expected_abundance = None, fname_out =
                      "tmp/infall_hod.cmass.hdf5", step = 323,
                      step_z=None, plot=False):
    if step_z is None:
        step2z = dtk.StepZ(sim_name='AlphaQ')
        step_z = step2z.get_z(step)
    mt_group = MergerTreeGroup(fname, num)
    # if expected_abundance is not None:
    #     vol = vol_per_num*num
    #     expected_count = vol*expected_abundance
    #     infall  = mt_group.get_infall_by_count(expected_count, step, min_mass=5e12)
    # else:
    #     infall = mt_group.get_infall_by_count(100*num, step)
    infall=1.11e13
    print("INFALL: ", infall)
    mass, count = mt_group.get_halo_mass_and_infall_counts(infall, step)

    mass_bins = dtk.get_logbins(mass, bins=32)
    mass_bins_cen  = dtk.bins_avg(mass_bins)
    avg = dtk.binned_average(mass, count, mass_bins)
    if plot:
        plt.figure()
        plt.loglog(mass, count, '.', alpha=0.3)
        plt.plot(mass_bins_cen, avg, '-r', label='average')
        plt.legend(loc='best')
        plt.xlabel('Halo FoF Mass [Msun/h, h=0.7]')
        plt.ylabel('Infall Counts')
        plt.title('Infall Mass: {:.2e}'.format(infall))
        plt.show()
    dtk.save_dict_hdf5(fname_out,{"mass_bins_cen":mass_bins_cen,
                                              "avg_cnt"      :avg})
def make_fof_cat(mass, count, x, y, z):
    fof_cat = {}
    fof_cat['fof_mass'] = mass
    fof_cat['galaxy_number'] = count
    fof_cat['x'] = x
    fof_cat['y'] = y
    fof_cat['z'] = z
    m180b, r180b, c180b = convert_step401_fofmass_to_m180b(mass, return_all=True)
    fof_cat['m180b'] = m180b
    fof_cat['r180b'] = r180b
    fof_cat['c180b'] = c180b
    fof_cat['htag']  = np.zeros_like(x, dtype=int)
    return fof_cat

def plot_galaxies(gal_cat):
    plt.figure()
    plt.plot(gal_cat['x'], gal_cat['y'], ',')
    
    plt.figure()
    plt.plot(gal_cat['x'], gal_cat['y'], ',')
    plt.show()

def get_SHAM_2pt(mt_fname, num=None, vol_per_num=None,
                 expected_abundance=None, step=323, step_z = None):
    if step_z is None:
        step2z = dtk.StepZ(sim_name='AlphaQ')
        step_z = step2z.get_z(step)
    if (mt_fname, num) not in get_SHAM_2pt._cache:
        mt_group = MergerTreeGroup(mt_fname, num)
        get_SHAM_2pt._cache[(mt_fname, num)] = mt_group
    else:
        mt_group = get_SHAM_2pt._cache[(mt_fname, num)]
    if expected_abundance is not None:
        vol = vol_per_num*num
        expected_count = vol*expected_abundance
        infall  = mt_group.get_infall_by_count(expected_count, step, min_mass=5e12)
    else:
        infall = mt_group.get_infall_by_count(100*num, step)
    # infall=1.11e13
    print("INFALL: ", infall)
    mass, count, x, y, z = mt_group.get_halo_mass_and_infall_counts(infall, step, return_position=True)
    fof_cat = make_fof_cat(mass, count, x, y, z)
    gal_cat = populate_halos_with_galaxies_with_NFW(fof_cat)
    
    compute_cmass_hod_2pt(gal_cat, 256.0, gal_label='AlphaQ SHAM[{:.1e}]'.format(infall))
   

get_SHAM_2pt._cache = {}
    
    
    



if __name__ == "__main__":
    # load_merger_trees('/media/luna1/dkorytov/data/AlphaQ/merger_trees3/AlphaQ.${num}.hdf5',
    #                   num=3, vol_per_num = 256*256,
    #                   expected_abundance=3.6e-4, fname_out =
    #                   "tmp/infall_hod.cmass.AQ.hdf5", 
    #                   plot=True)

    # load_merger_trees('/media/luna1/dkorytov/data/OuterRim/merger_trees/OR.${num}.hdf5',
    #                   num=3, vol_per_num = 300*300*250,
    #                   expected_abundance=3.6e-4,
    #                   fname_out = "tmp/infall_hod.cmass.OR.hdf5")

    get_SHAM_2pt('/media/luna1/dkorytov/data/AlphaQ/merger_trees3/AlphaQ.${num}.hdf5',
                 num=256, vol_per_num = 256*256,
                 expected_abundance=3.6e-4)
    get_SHAM_2pt('/media/luna1/dkorytov/data/AlphaQ/merger_trees3/AlphaQ.${num}.hdf5',
                 num=256, vol_per_num = 256*256,
                 expected_abundance=4.1e-4)
    plt.show()
    # get_SHAM_2pt('/media/luna1/dkorytov/data/OuterRim/merger_trees/OR.${num}.hdf5',
    #              num=3, vol_per_num = 256*256,
    #              expected_abundance=3.6e-4)





