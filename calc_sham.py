#!/usr/bin/env python2.7

from __future__ import print_function, division
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

class MergerTree:
    def __init__(self, fname=None):
        if fname is not None:
            self.load_hdf5(fname)

    def load_hdf5(self, fname):
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
        self.find_progenitors()

    def find_progenitors(self):
        srt = np.argsort(self.nodeIndex)
        desc_indexes = dtk.search_sorted(self.nodeIndex, self.desc_nodeIndex, sorter=srt)
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
                # self.print_progen(i)
                # raw_input()

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
                                   
            
    def get_halo_mass_and_infall_counts(self, mass_limit, step, last_isolated=False):
        step_slct = (self.step == step) & (self.mass > mass_limit)
        halo_indexes = np.where(step_slct)[0]
        total_count = 0
        masses = []
        counts  = []
        # print(halo_indexes)
        
        for i, halo_i in enumerate(halo_indexes):
            # print(i/len(halo_indexes))
            # print("getting infall count for ", halo_i)
            # print("\tmass: {:.2e}".format(self.mass[halo_i]))

            cnt = self.count_halo_infalls(halo_i, mass_limit)
            masses.append(self.mass[halo_i])
            counts.append(cnt)
            # print("\tcnt : ", cnt)
        masses = np.array(masses)
        counts = np.array(counts)
        # plt.figure()
        # plt.loglog(masses, counts, '.', alpha=0.3)
        # mass_bins = dtk.get_logbins(masses)
        # mass_bins_cen  = dtk.bins_avg(mass_bins)
        # avg = dtk.binned_average(masses, counts, mass_bins)
        # plt.plot(mass_bins_cen, avg, '-r')
        return masses, counts
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
            
    def get_halo_mass_and_infall_counts(self, infall, step):
        masses = []
        counts = []
        eta = dtk.ETA()
        mt_size = len(self.merger_trees)
        for i in range(0, mt_size):
            eta.print_progress(i, mt_size)
            m,c = self.merger_trees[i].get_halo_mass_and_infall_counts(infall, step)
            masses.append(m)
            counts.append(c)
        eta.print_done()
        masses = np.concatenate(masses)
        counts = np.concatenate(counts)
        return masses, counts

    def get_count_by_infall(self, infall, step):
        m,c = self.get_halo_mass_and_infall_counts(infall, step)
        return np.sum(c)


    def get_infall_by_count(self, target_count, step, min_mass=1e10, max_mass=1e16, tolerance=0.01, min_count = None, max_count=None):
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




def get_merger_tree_hod(fname, infall, step):
    print(fname)
    mt = MergerTree(fname)
    mass, cnts = mt.get_halo_mass_and_infall_counts(infall, 401)
    return mass, cnts

def load_merger_trees(fname, num=None):

    mt_group = MergerTreeGroup(fname, num)
    infall = mt_group.get_infall_by_count(100*num, 401)
    print("INFALL: ", infall)
    mass, count = mt_group.get_halo_mass_and_infall_counts(infall, 401)

    # eta = dtk.ETA()
    # if num is None:
    #     mass, count = get_merger_tree_hod(fname,infall,401)
    # else:
    #     masses = []
    #     counts = []

    #     for i in range(0, num):
    #         eta.print_progress(i,num)
    #         mass, cnts = get_merger_tree_hod(fname.replace('${num}', str(i)), infall, 401)
    #         masses.append(mass)
    #         counts.append(cnts)
    #         mass = np.concatenate(masses)
    #         count = np.concatenate(counts)
    #         print()
    # eta.print_done()
    plt.figure()
    plt.loglog(mass, count, '.', alpha=0.3)
    mass_bins = dtk.get_logbins(mass)
    mass_bins_cen  = dtk.bins_avg(mass_bins)
    avg = dtk.binned_average(mass, count, mass_bins)
    plt.plot(mass_bins_cen, avg, '-r', label='average')
    plt.legend(loc='best')
    plt.xlabel('Halo FoF Mass [Msun/h, h=0.7]')
    plt.ylabel('Infall Counts')
    plt.title('Infall Mass: {:.2e}'.format(infall))
    plt.show()



if __name__ == "__main__":
    load_merger_trees('/media/luna1/dkorytov/data/AlphaQ/merger_trees3/AlphaQ.${num}.hdf5', num=256)




