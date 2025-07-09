import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pylab as plt
import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append('../src/')
import dictionaries
import omnifoldHI

########################################################################
### ANALYSIS
########################################################################

# Config files
config_dict_data_file     = 'config_dict_data.json'
config_analysis_file_list = ['config_anl_mfHI3.json',
                             'config_anl_mfHI7.json',
                             'config_anl_mfHI12.json',
                             'config_anl_mfHI18.json']

# Run analysis for each config                           
for config_analysis_file in config_analysis_file_list:

    ####################################
    ## ANALYSIS CONFIG
    ####################################

    with open(config_analysis_file, 'r') as config_file:
        config_analysis = json.load(config_file)
        
        # Print analysis config
        print('\nANALYSIS CONFIG:')
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

    ####################################
    ## DATA DICTIONARY
    ####################################

    # Get config to build dictionaries
    with open(config_dict_data_file, 'r') as config_file:
        dict_data_empty = json.load(config_file)

        # Create dictionary from chosen data
        print('Creating dict_data.')
        dict_data = dictionaries.create_dict_data(dict_data_empty,
                                                  synthetic_cause_file, synthetic_effect_file,
                                                  nature_cause_file, nature_effect_file,
                                                  norm_use_F=False, verbose=False)

    ####################################
    ## OMNIFOLD-HI UNFOLDING
    ####################################

    # OmniFold-HI running      
    try:
        multifoldHI_out = dictionaries.load(f'multifoldHI_out_{code}.pkl')
    except:
        print('Unfolding w/ OmniFold-HI.')
        multifoldHI_out = omnifoldHI.multifold_unc_from_dict(dict_data, multifoldHI_obs,
                                                             n_iterations=multifoldHI_n_iterations,
                                                             unc_n_stat=multifoldHI_unc_n_stat,
                                                             unc_n_syst=multifoldHI_unc_n_syst)
        dictionaries.save(multifoldHI_out, f'multifoldHI_out_{code}.pkl')

    # Turn weights into histos
    try:
        dict_multifoldHI = dictionaries.load(f'dict_multifoldHI_{code}.pkl')
    except:
        print('Binning reweighed events.')
        dict_multifoldHI = omnifoldHI.create_dict_multifoldHI(dict_data,
                                                              multifoldHI_out['weights_stat'],
                                                              multifoldHI_out['weights_syst'],
                                                              multifold_it=-1,
                                                              norm_use_F=False)
        dictionaries.save(dict_multifoldHI, f'dict_multifoldHI_{code}.pkl')

########################################################################
### PLOT
########################################################################

# Function to plot OmniFold-HI histos
def plot_multifold (obs_dict_multifoldHI, obs_dict_data, axy, axr, axFTy, axFTr, style):

    cmap_mfHI     = plt.get_cmap('YlOrRd')
    hist_style['mfHI1']  = {'stat':{'color':cmap_mfHI(1.2/6), 'label':f'OmniFold-HI 1D', 'lw':1.5, 'capsize':2, 'zorder':1},
                            'syst':{'color':cmap_mfHI(1.2/6), 'alpha':.3}}
    hist_style['mfHI3']  = {'stat':{'color':cmap_mfHI(2/6), 'label':f'OmniFold-HI 3D', 'lw':1.5, 'capsize':2, 'zorder':2},
                            'syst':{'color':cmap_mfHI(2/6), 'alpha':.3}}
    hist_style['mfHI7']  = {'stat':{'color':cmap_mfHI(3/6), 'label':f'OmniFold-HI 7D', 'lw':1.5, 'capsize':2, 'zorder':3},
                            'syst':{'color':cmap_mfHI(3/6), 'alpha':.3}}
    hist_style['mfHI9']  = {'stat':{'color':cmap_mfHI(4/6), 'label':f'OmniFold-HI 9D', 'lw':1.5, 'capsize':2, 'zorder':4},
                            'syst':{'color':cmap_mfHI(4/6), 'alpha':.3}}
    hist_style['mfHI12'] = {'stat':{'color':cmap_mfHI(4/6), 'label':f'OmniFold-HI 12D', 'lw':1.5, 'capsize':2, 'zorder':5},
                            'syst':{'color':cmap_mfHI(4/6), 'alpha':.3}}
    hist_style['mfHI18'] = {'stat':{'color':cmap_mfHI(5/6), 'label':f'OmniFold-HI 18D', 'lw':1.5, 'capsize':2, 'zorder':6},
                            'syst':{'color':cmap_mfHI(5/6), 'alpha':.3}}

    # Plot unfolded
    if axy is not None:
        axy.errorbar(obs_dict_multifoldHI['midbin_C'], obs_dict_multifoldHI['hist_unf_C'], obs_dict_multifoldHI['hist_unf_C_unc_stat'], obs_dict_multifoldHI['midbin_C_unc'], fmt=',', **hist_style[style]['stat'])
        axy.bar(obs_dict_multifoldHI['midbin_C'], 2*obs_dict_multifoldHI['hist_unf_C_unc_stat'], 2*obs_dict_multifoldHI['midbin_C_unc'], obs_dict_multifoldHI['hist_unf_C']-obs_dict_multifoldHI['hist_unf_C_unc_stat'], **hist_style[style]['syst'])

    ###
    # Compute unfolded raito to truth
    omn_tru          = (obs_dict_multifoldHI['hist_unf_C']+1e-10)/(obs_dict_data['hist_nat_C']+1e-10)
    omn_tru_unc_stat = (obs_dict_multifoldHI['hist_unf_C_unc_stat'])/(obs_dict_data['hist_nat_C']+1e-10)
    omn_tru_unc_syst = (obs_dict_multifoldHI['hist_unf_C_unc_syst'])/(obs_dict_data['hist_nat_C']+1e-10)

    ###
    # Plot unfolded ratio to truth
    axr.errorbar(obs_dict_multifoldHI['midbin_C'], omn_tru, omn_tru_unc_stat, obs_dict_multifoldHI['midbin_C_unc'], **hist_style[style]['stat'])
    axr.bar(obs_dict_multifoldHI['midbin_C'], 2*omn_tru_unc_syst, 2*obs_dict_multifoldHI['midbin_C_unc'], omn_tru-omn_tru_unc_syst, **hist_style[style]['syst'])

    ###
    # Plot unfolded fake
    if axFTy is not None:
        axFTy.errorbar([1], obs_dict_multifoldHI['hist_unf_F'], obs_dict_multifoldHI['hist_unf_F_unc_stat'], [.4], **hist_style[style]['stat'])
        axFTy.bar([1], 2*obs_dict_multifoldHI['hist_unf_F_unc_stat'], [2*.4], obs_dict_multifoldHI['hist_unf_F']-obs_dict_multifoldHI['hist_unf_F_unc_stat'], **hist_style[style]['syst'])

    ###
    # Compute unfolded fake ratio to truth
    fake_omn_tru          = (obs_dict_multifoldHI['hist_unf_F']+1e-10)/(obs_dict_data['hist_nat_F']+1e-10)
    fake_omn_tru_unc_stat = (obs_dict_multifoldHI['hist_unf_F_unc_stat']+1e-10)/(obs_dict_data['hist_nat_F']+1e-10)
    fake_omn_tru_unc_syst = (obs_dict_multifoldHI['hist_unf_F_unc_syst']+1e-10)/(obs_dict_data['hist_nat_F']+1e-10)

    ###
    # Plot unfolded fake ratio to truth
    axFTr.errorbar([1], fake_omn_tru, fake_omn_tru_unc_stat, [.4], **hist_style[style]['stat'])
    axFTr.bar([1], 2*fake_omn_tru_unc_syst, [2*.4], fake_omn_tru-fake_omn_tru_unc_syst, **hist_style[style]['syst'])

    ###
    # Plot unfolded trash
    if axFTy is not None:
        axFTy.errorbar([2], obs_dict_multifoldHI['hist_unf_T'], obs_dict_multifoldHI['hist_unf_T_unc_stat'], [.4], **hist_style[style]['stat'])
        axFTy.bar([2], 2*obs_dict_multifoldHI['hist_unf_T_unc_stat'], [2*.4], obs_dict_multifoldHI['hist_unf_T']-obs_dict_multifoldHI['hist_unf_T_unc_stat'], **hist_style[style]['syst'])

    ###
    # Compute unfolded trash ratio to truth
    tras_omn_dat          = (obs_dict_multifoldHI['hist_unf_T']+1e-10)/(obs_dict_data['hist_nat_T']+1e-10)
    tras_omn_dat_unc_stat = (obs_dict_multifoldHI['hist_unf_T_unc_stat']+1e-10)/(obs_dict_data['hist_nat_T']+1e-10)
    tras_omn_dat_unc_syst = (obs_dict_multifoldHI['hist_unf_T_unc_syst']+1e-10)/(obs_dict_data['hist_nat_T']+1e-10)

    ###
    # Plot unfolded trash ratio to truth
    axFTr.errorbar([2], tras_omn_dat, tras_omn_dat_unc_stat, [.4], **hist_style[style]['stat'])
    axFTr.bar([2], 2*tras_omn_dat_unc_syst, [2*.4], tras_omn_dat-tras_omn_dat_unc_syst, **hist_style[style]['syst'])

    return

####################################
## LOAD DICTIONARIES
####################################
dict_multifoldHI_mfHI3  = dictionaries.load('dict_multifoldHI_mfHI3.pkl')
dict_multifoldHI_mfHI7  = dictionaries.load('dict_multifoldHI_mfHI7.pkl')
dict_multifoldHI_mfHI12 = dictionaries.load('dict_multifoldHI_mfHI12.pkl')
dict_multifoldHI_mfHI18 = dictionaries.load('dict_multifoldHI_mfHI18.pkl')

####################################
## PLOTTING
####################################

# Observables to plot
plot_obs = ["pt_jet", "eta_jet", "phi_jet", "m_jet", "n_jet", "z_DyG", "th_DyG", "kt_DyG", "pt_SD", "m_SD", "z_SD", "pt_RSD", "n_RSD", "tau1", "tau2", "tau3", "tau4", "tau5"]

with PdfPages(f'../figures/omnifoldHI_pages.pdf') as pdf:
    with mpl.rc_context({'font.family': 'sans-serif', 'font.size': 14, 'text.usetex':False}):

        hist_style = {
            'gen': {'label':'gen', 'lw':2, 'color':'royalblue'},
            'sim': {'label':'sim', 'lw':2, 'color':'orange'},
            'tru': {'label':'Truth', 'lw':2, 'color':'green'},
            'dat': {'label':'dat', 'lw':2, 'color':'k'}
        }

        # Plot for each observable
        figs = []
        axes = []
        for i, key in enumerate(plot_obs):

            obs_dict_data = dict_data[key]

            fig = plt.figure(figsize=(5,5))

            gs = fig.add_gridspec(2,2, height_ratios=(3,1), width_ratios=(5,1), hspace=0, wspace=.03)

            axy   = fig.add_subplot(gs[0,0])
            axr   = fig.add_subplot(gs[1,0], sharex=axy)
            axFTy = fig.add_subplot(gs[0,1])
            axFTr = fig.add_subplot(gs[1,1], sharex=axFTy, sharey=axr)

            figs.append(fig)
            axes.append({'axy':axy, 'axr':axr, 'axFTy':axFTy, 'axFTr':axFTr})

            # BASELINE
            ## Plot datasets
            axy.stairs(obs_dict_data['hist_nat_E'], obs_dict_data['bins_E'], baseline=0, fill=True, alpha=.2, color='k', label='Measured')
            axy.stairs(obs_dict_data['hist_nat_C'], obs_dict_data['bins_C'], baseline=0, fill=True, alpha=.4, color='green', label='Truth')

            ###
            # Plot fake
            axFTy.stairs(obs_dict_data['hist_nat_F'], [.6,1.4], baseline=0, fill=True, alpha=.4, color='green')

            ###
            # Plot trash
            axFTy.stairs(obs_dict_data['hist_nat_T'], [1.6,2.4], baseline=0, fill=True, alpha=.2, color='k')

            ###
            # Compute raito to truth
            tru_tru     = (obs_dict_data['hist_nat_C']+1e-10)/(obs_dict_data['hist_nat_C']+1e-10)
            tru_tru_std = (obs_dict_data['hist_nat_C_unc'])/(obs_dict_data['hist_nat_C']+1e-10)

            ###
            # Plot ratio to truth
            axr.errorbar(obs_dict_data['midbin_C'], np.ones(len(obs_dict_data['bins_C'])-1), 0,obs_dict_data['midbin_C_unc'], **hist_style['tru'])
            axr.stairs(tru_tru+tru_tru_std, obs_dict_data['bins_C'], baseline=1, fill=True, alpha=.4, color='green')
            axr.stairs(tru_tru-tru_tru_std, obs_dict_data['bins_C'], baseline=1, fill=True, alpha=.4, color='green')

            ###
            # Compute fake ratio to truth
            fake_tru_tru     = (obs_dict_data['hist_nat_F']+1e-10)/(obs_dict_data['hist_nat_F']+1e-10)
            fake_tru_tru_std = (obs_dict_data['hist_nat_F_unc']+1e-10)/(obs_dict_data['hist_nat_F']+1e-10)

            ###
            # Plot fake ratio to truth
            axFTr.errorbar([1], [1], 0, [.4], **hist_style['tru'])
            axFTr.stairs(fake_tru_tru-fake_tru_tru_std, [.6,1.4], baseline=1, fill=True, alpha=.4, color='green')
            axFTr.stairs(fake_tru_tru+fake_tru_tru_std, [.6,1.4], baseline=1, fill=True, alpha=.4, color='green')

            ###
            # Compute trash ratio to truth
            tras_dat_dat     = (obs_dict_data['hist_nat_T']+1e-10)/(obs_dict_data['hist_nat_T']+1e-10)
            tras_dat_dat_std = (obs_dict_data['hist_nat_T_unc']+1e-10)/(obs_dict_data['hist_nat_T']+1e-10)

            ###
            # Plot trash ratio to truth
            axFTr.errorbar([2], [1], 0, [.4], **hist_style['dat'])
            axFTr.stairs(fake_tru_tru-fake_tru_tru_std, [1.6,2.4], baseline=1, fill=True, alpha=.2, color='k')
            axFTr.stairs(fake_tru_tru+fake_tru_tru_std, [1.6,2.4], baseline=1, fill=True, alpha=.2, color='k')

            # MULTIFOLD 3D
            obs_dict_multifoldHI = dict_multifoldHI_mfHI3[key]
            plot_multifold(obs_dict_multifoldHI, obs_dict_data, axy, axr, axFTy, axFTr, style='mfHI3')

            # MULTIFOLD 7D
            obs_dict_multifoldHI = dict_multifoldHI_mfHI7[key]
            plot_multifold(obs_dict_multifoldHI, obs_dict_data, axy, axr, axFTy, axFTr, style='mfHI7')

            # MULTIFOLD 12D
            obs_dict_multifoldHI = dict_multifoldHI_mfHI12[key]
            plot_multifold(obs_dict_multifoldHI, obs_dict_data, axy, axr, axFTy, axFTr, style='mfHI12')

            # MULTIFOLD 18D
            obs_dict_multifoldHI = dict_multifoldHI_mfHI18[key]
            plot_multifold(obs_dict_multifoldHI, obs_dict_data, axy, axr, axFTy, axFTr, style='mfHI18')

            # Labels and limits
            axr.set_xlabel(obs_dict_data['xlabel'])
            axy.set_ylabel(obs_dict_data['ylabel'])
            axr.set_ylabel('ratio to\ntruth')
            axFTr.set_ylim(.82,1.18)
            axFTr.set_xlim(.3,2.7)
            
            # Ticks
            axFTr.set_xticks(ticks=[1,2], labels=[r'$F$',r'$T$'])
            axFTr.set_yticks(ticks=[.9,1,1.1])
            for axis in [axy,axr,axFTy,axFTr]:
                axis.minorticks_on()
                axis.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            for axis in [axFTy, axFTr]:
                axis.tick_params(axis='x', which='minor', top=False, bottom=False)
            for axis in [axy, axFTy]:
                axis.tick_params(axis='x', labelbottom=False)
            for axis in [axFTy]:
                axis.tick_params(axis='y', labelleft=False)
                axis.tick_params(axis='y', labelright=True)
            for axis in [axFTr]:
                axis.tick_params(axis='y', labelleft=False)
                axis.tick_params(axis='y', labelright=False)

        axes[14]['axFTy'].text(2.5,1, f'Truth/Measured: Herwig7 (+ Bkg. + Delphes)', color='gray', fontsize='small', ha='right', va='top', transform=axes[14]['axFTy'].transAxes, rotation=-90)
        axes[14]['axFTy'].text(2.1,1, f'Generated/Simulated: Pythia8 (+ Bkg. + Delphes)', color='gray', fontsize='small', ha='right', va='top', transform=axes[14]['axFTy'].transAxes, rotation=-90)

        axes[7]['axy'].text(.11,.96, r'hardest jet in pp@5.02 TeV', fontsize='small', ha='left', va='top', transform=axes[7]['axy'].transAxes)
        axes[7]['axy'].text(.11,.89, r'$R_{ak_t}=0.4, |\eta|<2.8$', fontsize='small', ha='left', va='top', transform=axes[7]['axy'].transAxes)
        axes[7]['axy'].text(.11,.81, r'$p_T>250$ GeV', fontsize='small', ha='left', va='top', transform=axes[7]['axy'].transAxes)

        axes[0]['axy'].set_xlim(dict_data[plot_obs[0]]['bins_C'][0],dict_data[plot_obs[0]]['bins_C'][-1])
        axes[1]['axy'].set_xlim(dict_data[plot_obs[1]]['bins_C'][0],dict_data[plot_obs[1]]['bins_C'][-1])
        axes[2]['axy'].set_xlim(dict_data[plot_obs[2]]['bins_C'][0],dict_data[plot_obs[2]]['bins_C'][-1])
        
        #axes[0]['axr'].set_xticks(np.linspace(250,650,3))

        axes[0]['axy'].legend(frameon=False, fontsize='small')

        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


print('\nDone.')