import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pylab as plt
import argparse
import json

import sys
sys.path.append('../src/')
import dictionaries
import omnifoldHI

# Get configuraion from config_file
parser = argparse.ArgumentParser(description='Unfold analysis based on config file')
parser.add_argument('-c_data', '--config_dict_data_file', type=str, help='Data dictionary configuration file.')
parser.add_argument('-c_anl' , '--config_analysis_file' , type=str, help='Analysis configuration file.')
args = parser.parse_args()

# Get config to perform analysis
with open(args.config_analysis_file, 'r') as config_file:
    config_analysis = json.load(config_file)
    
    # Print analysis config
    print('ANALYSIS CONFIG:')
    for key, value in config_analysis.items():
        print(f' > {key}: {value}')

    # Analysis code
    code = config_analysis['code']

    # Choose natural and synthetic data files
    synthetic_cause_file  = config_analysis['synthetic_cause_file']
    synthetic_effect_file = config_analysis['synthetic_effect_file']
    nature_cause_file     = config_analysis['nature_cause_file']
    nature_effect_file    = config_analysis['nature_effect_file']

    # Observables to unfold
    multifoldHI_obs = config_analysis['multifoldHI_obs']

    # MultiFoldHI setup
    multifoldHI_n_iterations = config_analysis['multifoldHI_n_iterations']
    multifoldHI_unc_n_stat   = config_analysis['multifoldHI_unc_n_stat']
    multifoldHI_unc_n_syst   = config_analysis['multifoldHI_unc_n_syst']

# Get config to build dictionaries
with open(args.config_dict_data_file, 'r') as config_file:
    dict_data_empty = json.load(config_file)

    # Create dictionary from chosen data
    print('\nCreating dict_data.')
    dict_data = dictionaries.create_dict_data(dict_data_empty,
                                              synthetic_cause_file, synthetic_effect_file,
                                              nature_cause_file, nature_effect_file,
                                              norm_use_F=False, verbose=False)

# Save dict_data
data_code = f'{code.split("_")[0]}_{code.split("_")[1]}'
dictionaries.save(dict_data, f'dict_data_{data_code}.pkl')

# Unfold with MultiFoldHI                       
print('\nUnfolding w/ MultiFold-HI.')
multifoldHI_out = omnifoldHI.multifold_unc_from_dict(dict_data, multifoldHI_obs,
                                                     n_iterations=multifoldHI_n_iterations,
                                                     unc_n_stat=multifoldHI_unc_n_stat,
                                                     unc_n_syst=multifoldHI_unc_n_syst)
dict_multifoldHI = omnifoldHI.create_dict_multifoldHI(dict_data,
                                                      multifoldHI_out['weights_stat'],
                                                      multifoldHI_out['weights_syst'],
                                                      multifold_it=-1,
                                                      norm_use_F=False)

# Save dictionaries 
print('\nSaving dictionaires.')
dictionaries.save(multifoldHI_out, f'multifoldHI_out_{code}.pkl')
dictionaries.save(dict_multifoldHI, f'dict_multifoldHI_{code}.pkl')
print('\nDone.')
