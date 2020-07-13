import numpy as np
import logging
import h5py
import glob
import six
import os
import sys
import argparse
import pandas as pd
from astropy.cosmology import LambdaCDM

h = 0.6774
cosmo = LambdaCDM(H0=100*h, Om0=0.3089, Ob0=0.0486, Ode0=0.6911, Tcmb0=2.73)


"""

Illustris Public Script helper functions

"""
def maxPastMass(tree, index, partType='stars'):
    """ Get maximum past mass (of the given partType) along the main branch of a subhalo
        specified by index within this tree. """
    ptNum = partTypeNum(partType)

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index] + 1
    masses = tree['SubhaloMassType'][index: index + branchSize, ptNum]
    return np.max(masses)

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)
        
    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    
    raise Exception("Unknown particle type name.")

cat = np.load('snapTNG.npy')
gys = cat[2]
snapshots= cat[0]





def get_current_mass(tree, index, partType='stars'):

    ptNum = partTypeNum(partType)

    return tree['SubhaloMassType'][index, ptNum]

def numMergers(tree, minMassRatio=1e-01, massPartType='stars', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    central_snapshots = np.array([])
    central_subhalos = np.array([])

    merger_next_progenitors = np.array([])
    merger_first_progenitors = np.array([])
    merger_ratios = np.array([])
    fp_log_masses = np.array([])
    np_log_masses = np.array([])

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]

    current_central_snapshot = tree['SnapNum'][index]
    current_central_subhalo = tree['SubfindID'][index]

    while fpID != -1:

        fpIndex = index + (fpID - rootID)
        fpMass  = maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = maxPastMass(tree, npIndex, massPartType)
            
            np_log_mass = to_log_mass(npMass)
            fp_log_mass = to_log_mass(fpMass)

            mass_limit = 8.0

            if fp_log_mass > mass_limit or np_log_mass > mass_limit:
            
                ratio = npMass / fpMass
                
                if ratio >= minMassRatio and ratio <= invMassRatio:
                    numMergers += 1

                    central_snapshots = np.append(central_snapshots, current_central_snapshot)
                    central_subhalos = np.append(central_subhalos, current_central_subhalo)

                    merger_next_progenitors = np.append(merger_next_progenitors, npIndex)
                    merger_first_progenitors= np.append(merger_first_progenitors, fpIndex)
                    merger_ratios = np.append(merger_ratios, ratio)
                    fp_log_masses = np.append(fp_log_masses, fp_log_mass)
                    np_log_masses = np.append(np_log_masses, np_log_mass)

            npID = tree['NextProgenitorID'][npIndex]
	
	
        fpID = tree['FirstProgenitorID'][fpIndex]
        current_central_subhalo = tree['SubfindID'][fpIndex]
        current_central_snapshot = tree['SnapNum'][fpIndex]
        
    return (numMergers, merger_first_progenitors.astype(int), merger_ratios, fp_log_masses, np_log_masses, merger_next_progenitors.astype(int), central_snapshots, central_subhalos)

def to_log_mass(illustris_mass):
    return np.round(np.log10(illustris_mass * 1e10 / cosmo.h), 2) 


def find_matching_snapshots(offset=0.3):

    snapshots_to_match = np.array([])

    for i, gy in enumerate(gys):
         diff = gys - gy
         idx = np.where(abs(diff) < offset)
         snapshots_to_match = np.append(snapshots_to_match, snapshots[idx])

    return set(snapshots_to_match)

def match_snapshots(snaps_from_subhalo, search_offset=0.3):
    
    matched_halos_idxs = np.array([])
    
    matching_snapshots = find_matching_snapshots(offset=search_offset)
    for i, s in enumerate(snaps_from_subhalo):
        if(s in matching_snapshots):
            matched_halos_idxs = np.append(matched_halos_idxs, i)
     
    return matched_halos_idxs


def merger_finder(tree, tree_number, timescale=0.6):
    
    subhalosIDs = np.where(tree['SnapNum'][:] == 99)[0].astype(int)
    size = len(subhalosIDs)    
    merger_output = open('merger_run{}.dat'.format(tree_number), 'a+')
        
    for j, id in enumerate(subhalosIDs):
        num, subhalos, ratios, fp_masses, np_masses, merger_next_progenitors, central_snapshots, central_subhalos  = numMergers(tree, index=id)
        
        if(num > 0):
            snaps = np.array(tree['SnapNum'][:])[subhalos]

            matches = match_snapshots(snaps, search_offset=timescale)
            matches = matches.astype(int)

            logging.debug('Tree{}-{}% {}/{} Climbing {} subhalo tree. Found {} mergers. {} are snapshot matched'.format(tree_number, np.round(((j+1)/size)*100,1), j+1, size, id, num, len(matches)))

            if(len(matches)):     

                subfinds = np.array(tree['SubfindID'][:])[subhalos[matches]]
                mnps = np.array(tree['SubfindID'][:])[merger_next_progenitors[matches]]
                
                for sf, snp, ratio, fp_mass, np_mass, mnp, csp, csb in zip(subfinds, snaps[matches], ratios, fp_masses, np_masses, mnps, central_snapshots[matches], central_subhalos[matches]):
                    merger_output.write('{};{};{};{};{};{};{};{};{};{};{}\n'.format(snp, sf, mnp, tree['SubfindID'][id], np.round(ratio, 4), fp_mass, np_mass, csp, csb, num, len(matches)))
    
            else:
                logging.debug('No available mocks for mergers of this Subhalo')    

            merger_output.flush()
            os.fsync(merger_output.fileno())


def __handle_input(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('tree_file',
                        help='Path to the sublink merger tree file'
                        )
    
    parser.add_argument('--parallel', nargs='?', default=False, const=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = __handle_input(sys.argv)
    print(args.tree_file)
    try:
        tree = h5py.File(args.tree_file, 'r')

        #assume tree file has tree_extended.num.hdf5 name
        num = args.tree_file.split('.')[1]

        logging.basicConfig(filename=f'tireless-tracker-{num}.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

        merger_finder(tree, tree_number=num)

        logging.info('Run complete.')
    except FileNotFoundError:
        logging.error('Fits not found with path {}.\
               Please try again with a different path'.format(args.trees_folder))