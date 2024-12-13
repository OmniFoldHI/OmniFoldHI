import numpy as np
import argparse
import pickle
import copy

def read_cause_effect (datafile_cause, datafile_effect):

    def get_labels (filename):
        file = open(filename,'r')
        labels_line = file.readline().strip('#').split()
        file.close()
        return [ label.strip(',') for label in labels_line if '[' not in label ]

    dataset = {}

    # add cause events
    cause_labels = get_labels(datafile_cause)

    cause_values = np.loadtxt(datafile_cause)
    for i,label in enumerate(cause_labels):
        dataset['cause_'+label] = cause_values.T[i]
        
    # add raw evs
    effect_labels = get_labels(datafile_effect)
    effect_values = np.loadtxt(datafile_effect)
    for i,label in enumerate(effect_labels):
        dataset['effect_'+label] = effect_values.T[i]

    return dataset

def associate_data (obs_dict: dict, synthetic: dict, nature: dict, verbose=True):

    # Calculate quantities to be stored in obs
    for label, obs in obs_dict.items():
        
        # calculate obs for GEN, SIM, DATA, and TRUE
        obs['evs_syn_C'], obs['evs_syn_E'] = obs['func'](synthetic, 'cause'), obs['func'](synthetic, 'effect')
        obs['evs_nat_C'], obs['evs_nat_E'] = obs['func'](nature   , 'cause'), obs['func'](nature   , 'effect')

        # remove 'func' to be able to export with pickle
        obs['func'] = None
        
        # setup bins
        obs['bins_C']       = np.linspace(obs['xlim_C'][0], obs['xlim_C'][1], obs['nbins_C']+1)
        obs['bins_E']       = np.linspace(obs['xlim_E'][0], obs['xlim_E'][1], obs['nbins_E']+1)
        obs['midbin_C']     = (obs['bins_C'][:-1] + obs['bins_C' ][1:])/2
        obs['midbin_E']     = (obs['bins_E'][:-1] + obs['bins_E'][1:])/2
        obs['binwidth_C']   = obs['bins_C'][1] - obs['bins_C'][0]  # assuming linear binning
        obs['binwidth_E']   = obs['bins_E'][1] - obs['bins_E'][0]
        obs['midbin_C_unc'] = obs['binwidth_C']/2  # assuming linear binning
        obs['midbin_E_unc'] = obs['binwidth_E']/2

        # Event number (for normalization)
        obs['nevs_syn_C'] = np.count_nonzero(obs['evs_syn_C'])
        obs['nevs_syn_E'] = np.count_nonzero(obs['evs_syn_E'][obs['evs_syn_E']!=obs['val_T']])
        obs['nevs_nat_C'] = np.count_nonzero(obs['evs_nat_C'])
        obs['nevs_nat_E'] = np.count_nonzero(obs['evs_nat_E'][obs['evs_nat_E']!=obs['val_T']])

        # get the histograms
        obs['hist_syn_C'] = np.histogram(obs['evs_syn_C'], bins=obs['bins_C'] )[0] / obs['binwidth_C'] / obs['nevs_syn_C']
        obs['hist_syn_E'] = np.histogram(obs['evs_syn_E'], bins=obs['bins_E'] )[0] / obs['binwidth_E'] / obs['nevs_syn_E']
        obs['hist_nat_C'] = np.histogram(obs['evs_nat_C'], bins=obs['bins_C'] )[0] / obs['binwidth_C'] / obs['nevs_nat_C']
        obs['hist_nat_E'] = np.histogram(obs['evs_nat_E'], bins=obs['bins_E'] )[0] / obs['binwidth_E'] / obs['nevs_nat_E']

        # get the standard deviations
        obs['hist_syn_C_unc'] = np.sqrt(np.histogram(obs['evs_syn_C'], bins=obs['bins_C'] )[0]) / obs['binwidth_C'] / obs['nevs_syn_C']
        obs['hist_syn_E_unc'] = np.sqrt(np.histogram(obs['evs_syn_E'], bins=obs['bins_E'] )[0]) / obs['binwidth_E'] / obs['nevs_syn_E']
        obs['hist_nat_C_unc'] = np.sqrt(np.histogram(obs['evs_nat_C'], bins=obs['bins_C'] )[0]) / obs['binwidth_C'] / obs['nevs_nat_C']
        obs['hist_nat_E_unc'] = np.sqrt(np.histogram(obs['evs_nat_E'], bins=obs['bins_E'] )[0]) / obs['binwidth_E'] / obs['nevs_nat_E']

        # get fakes and trash
        obs['hist_syn_F'] = np.histogram( obs['evs_syn_C'], bins=[obs['val_F']-.01,obs['val_F']+.01] )[0] / obs['nevs_syn_C']
        obs['hist_syn_T'] = np.histogram( obs['evs_syn_E'], bins=[obs['val_T']-.01,obs['val_T']+.01] )[0] / obs['nevs_syn_E']
        obs['hist_nat_F'] = np.histogram( obs['evs_nat_C'], bins=[obs['val_F']-.01,obs['val_F']+.01] )[0] / obs['nevs_nat_C']
        obs['hist_nat_T'] = np.histogram( obs['evs_nat_E'], bins=[obs['val_T']-.01,obs['val_T']+.01] )[0] / obs['nevs_nat_E']

        # get fakes and trash standard deviations
        obs['hist_syn_F_unc'] = np.sqrt( np.histogram( obs['evs_syn_C'], bins=[obs['val_F']-.01,obs['val_F']+.01] )[0] ) / obs['nevs_syn_C']
        obs['hist_syn_T_unc'] = np.sqrt( np.histogram( obs['evs_syn_E'], bins=[obs['val_T']-.01,obs['val_T']+.01] )[0] ) / obs['nevs_syn_E']
        obs['hist_nat_F_unc'] = np.sqrt( np.histogram( obs['evs_nat_C'], bins=[obs['val_F']-.01,obs['val_F']+.01] )[0] ) / obs['nevs_nat_C']
        obs['hist_nat_T_unc'] = np.sqrt( np.histogram( obs['evs_nat_E'], bins=[obs['val_T']-.01,obs['val_T']+.01] )[0] ) / obs['nevs_nat_E']
        
        if verbose:
            print(label,'in dict_data.')

    return

def create_dict_data (dict_data_empty, synthetic_cause_file, synthetic_effect_file, nature_cause_file, nature_effect_file, verbose=True):

    synthetic = read_cause_effect( synthetic_cause_file, synthetic_effect_file )
    nature    = read_cause_effect( nature_cause_file   , nature_effect_file )

    # Create dictionary w/ observables and related info
    dict_data = copy.deepcopy(dict_data_empty)
    for obs_label, obs_dict in dict_data.items():
        obs_dict.update({'func': lambda dset, ptype, obs_label=obs_label: dset[ptype+f'_{obs_label}']})  # By adding obs_label=obs_label as a default argument to the lambda function, you capture the current value of obs_label at each iteration. This ensures that each lambda function has its own copy of obs_label with the value it had during that specific iteration.

    # Add data to dictionary
    associate_data(dict_data, synthetic, nature, verbose=verbose)

    return dict_data

def save (dictionary, dictionary_filename):

    with open(dictionary_filename, 'wb') as pkl_file:
        pickle.dump(dictionary, pkl_file)

    print('Pickled:', dictionary_filename)

    return

def load (dictionary_filename):

    with open(dictionary_filename,'rb') as pkl_file:
        dictionary = pickle.load(pkl_file)

    print('Unpickled:', dictionary_filename)

    return dictionary

