import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import json
from matplotlib.backends.backend_pdf import PdfPages

def MChistogram(data, bins, w=None, density=False, probability=False, dividebin=True):
    ''' Histogram of 1d data: dN/dbin. 
        It also returns with uncertainty assuming Normal distribution
        usual for event generator plots.
       
        Parameters
        ----------
        data : array, input data.
        bins : array, bin edges.
        w    : array, weight of each entry.
        density     : bool, normalizing the histogram to 1.
        probability : bool, dividing by the number of events.
        dividebin   : bool, dividing by the binsize.
              
        Returns
        -------
        [y, yerr, bins]
        
        Use plt.errorbar(bins[:-1]+np.diff(bins)/2,y,yerr) for plotting.
    '''
    dx = np.diff(bins)
    if w is None: w = np.ones(len(data))
    y     = np.histogram(data, bins=bins, weights=w,    density=False)[0]
    y_err = np.histogram(data, bins=bins, weights=w**2, density=False)[0]
    if density==True: 
        return [y/sum(y)/dx, np.sqrt(y_err)/sum(y)/dx, bins]
    if probability==True:
        return [y/sum(w)/dx, np.sqrt(y_err)/sum(w)/dx, bins]
    if dividebin==True:
        return [y/dx,        np.sqrt(y_err)/dx,        bins]
    return [y, np.sqrt(y_err), bins] 

# Load dataset
print('Loading datasets.')
p8_c = np.loadtxt("../../data/Pythia_bkg0_Delphes_Generated.out")
p8_e = np.loadtxt("../../data/Pythia_bkg1_Delphes_Simulated.out")
h7_c = np.loadtxt("../../data/Herwig_bkg0_Delphes_Generated.out")
h7_e = np.loadtxt("../../data/Herwig_bkg1_Delphes_Simulated.out")

# Load observable format
with open('../config_dict_data.json') as f:
    obs = json.load(f)

nature_label = 'Herwig7 (+ Bkg. + Delphes)'
synthetic_label = 'Pythia8 (+ Bkg. + Delphes)'

print('Plotting.')
with mpl.rc_context({'font.family': 'sans-serif', 'font.size': 14, 'text.usetex':False}):

    hist_style = {
        'tru': {'label':'Truth', 'lw':2, 'color':'green'},
        'dat': {'label':'dat', 'lw':2, 'color':'k'}
    }

    with PdfPages('data.pdf') as pdf:

        for i,key in enumerate(obs):
            print(key)
            # if key not in ['pt_jet','kt_DyG','tau2']: continue

            # Data out of binrange is trash
            p8_c_clean = np.array([val if (val!=-1 and obs[key]['xlim_C'][0]<=val<=obs[key]['xlim_C'][1]) else -7 for val in p8_c.T[i]])
            p8_e_clean = np.array([val if (val!=-1 and obs[key]['xlim_E'][0]<=val<=obs[key]['xlim_E'][1]) else -7 for val in p8_e.T[i]])
            h7_c_clean = np.array([val if (val!=-1 and obs[key]['xlim_C'][0]<=val<=obs[key]['xlim_C'][1]) else -7 for val in h7_c.T[i]])
            h7_e_clean = np.array([val if (val!=-1 and obs[key]['xlim_E'][0]<=val<=obs[key]['xlim_E'][1]) else -7 for val in h7_e.T[i]])

            # Histograms
            xbins_c = np.linspace(*obs[key]['xlim_C'], obs[key]['nbins_C'])
            h_p8_c = MChistogram(p8_c_clean, xbins_c, density=True)
            h_h7_c = MChistogram(h7_c_clean, xbins_c, density=True)
            xbins_e = np.linspace(*obs[key]['xlim_E'], obs[key]['nbins_E'])
            h_p8_e = MChistogram(p8_e_clean, xbins_e, density=True)
            h_h7_e = MChistogram(h7_e_clean, xbins_e, density=True)

            # Fakes and trash
            h_p8_F = MChistogram(p8_c_clean, bins=[-7.1,-6.9], w=np.ones(len(p8_c_clean))/len(p8_c_clean))
            h_p8_T = MChistogram(p8_e_clean, bins=[-7.1,-6.9], w=np.ones(len(p8_e_clean))/len(p8_e_clean))
            h_h7_F = MChistogram(h7_c_clean, bins=[-7.1,-6.9], w=np.ones(len(h7_c_clean))/len(h7_c_clean))
            h_h7_T = MChistogram(h7_e_clean, bins=[-7.1,-6.9], w=np.ones(len(h7_e_clean))/len(h7_e_clean))

            # Create figure and grid
            fig, ax = plt.subplots(2,2, figsize=(5,5), height_ratios=(3,1), width_ratios=(5,1), gridspec_kw={'hspace': 0, 'wspace': 0.03})

            # Plot BASELINE
            # Datasets
            ax[0,0].stairs(h_h7_e[0], h_h7_e[-1], baseline=0, fill=True, alpha=.2, color='k', label='Measured')
            ax[0,0].stairs(h_h7_c[0], h_h7_c[-1], baseline=0, fill=True, alpha=.4, color='green', label='Truth')
            # Fake and trash probability
            ax[0,1].stairs(h_h7_F[0]*np.diff(h_h7_F[-1]), [.6,1.4], baseline=0, fill=True, alpha=.4, color='green')
            ax[0,1].stairs(h_h7_T[0]*np.diff(h_h7_T[-1]), [1.6,2.4], baseline=0, fill=True, alpha=.2, color='k')
            # Ratio to truth
            midbin_c = xbins_c[:-1] + np.diff(xbins_c)/2
            ax[1,0].plot(xbins_c, xbins_c/xbins_c, 'k:', lw=1.5)
            ax[1,0].stairs((h_h7_c[0]+h_h7_c[1])/(h_h7_c[0]+1e-20), h_h7_c[-1], baseline=(h_h7_c[0]-h_h7_c[1])/(h_h7_c[0]+1e-20), fill=True, alpha=.4, color='green')
            # Fake and trash ratio to truth
            ax[1,1].plot([0,3],[1,1],'k:',lw=1.5)
            # ax[1,1].errorbar([1], [1], 0, [.4], **hist_style['tru'])
            # ax[1,1].errorbar([2], [1], 0, [.4], **hist_style['dat'])
            ax[1,1].stairs((h_h7_F[0]+h_h7_F[1])/(h_h7_F[0]+1e-20), [.6,1.4], baseline=(h_h7_F[0]-h_h7_F[1])/(h_h7_F[0]+1e-20), fill=True, alpha=.4, color='green')
            ax[1,1].stairs((h_h7_T[0]+h_h7_T[1])/(h_h7_T[0]+1e-20), [1.6,2.4], baseline=(h_h7_T[0]-h_h7_T[1])/(h_h7_T[0]+1e-20), fill=True, alpha=.2, color='k')

            # Plot PRIOR
            # Datasets
            ax[0,0].stairs(h_p8_e[0], h_p8_e[-1], baseline=0, lw=2, color='k', label='Simulated')
            ax[0,0].stairs(h_p8_c[0], h_p8_c[-1], baseline=0, lw=2, color='green', label='Generated')
            # Fake and trash probability
            ax[0,1].stairs(h_p8_F[0]*np.diff(h_p8_F[-1]), [.6,1.4], baseline=0, **hist_style['tru'])
            ax[0,1].stairs(h_p8_T[0]*np.diff(h_p8_F[-1]), [1.6,2.4], baseline=0, **hist_style['dat'])
            # Ratio to truth
            ax[1,0].errorbar(midbin_c, h_p8_c[0]/(h_h7_c[0]+1e-20), h_p8_c[1]/(h_h7_c[0]+1e-20), np.diff(xbins_c)/2, **hist_style['tru'], fmt=',')
            ax[1,0].stairs(h_p8_c[0]/(h_h7_c[0]+1e-20), h_p8_c[-1], **hist_style['tru'])
            # Fake and trash ratio to truth
            ax[1,1].errorbar(1, h_p8_F[0]/(h_h7_F[0]+1e-20), h_p8_F[1]/(h_h7_F[0]+1e-20), 0.4, **hist_style['tru'], fmt=',')
            ax[1,1].errorbar(2, h_p8_T[0]/(h_h7_T[0]+1e-20), h_p8_T[1]/(h_h7_T[0]+1e-20), 0.4, **hist_style['dat'], fmt=',')

            # Labels and limits
            ax[1,0].set_xlabel(obs[key]['xlabel'])
            ax[0,0].set_ylabel(obs[key]['ylabel'])
            ax[1,0].set_ylabel('ratio to\ntruth')
            ax[0,0].set_xlim(*obs[key]['xlim_C'])
            ax[1,0].set_xlim(*obs[key]['xlim_C'])
            ax[1,0].set_ylim(.65,1.35)
            ax[0,1].set_ylim(0,1)
            ax[0,1].set_xlim(.3,2.7)
            ax[1,1].set_xlim(.3,2.7)
            ax[1,1].set_ylim(.65,1.35)
        
            # Ticks
            ax[0,1].set_xticks(ticks=[1,2])
            ax[1,1].set_xticks(ticks=[1,2], labels=[r'$F$',r'$T$'])
            ax[1,0].set_yticks(ticks=[.8,1,1.2])
            ax[1,1].set_yticks(ticks=[.8,1,1.2])
            ax[0,0].minorticks_on()
            ax[0,1].minorticks_on()
            ax[1,0].minorticks_on()
            ax[1,1].minorticks_on()
            ax[0,0].tick_params(axis='both', which='both', direction='in', top=True, right=True, labelbottom=False)
            ax[0,1].tick_params(axis='both', which='both', direction='in', top=True, right=True, labelbottom=False, labelleft=False, labelright=True)
            ax[1,0].tick_params(axis='both', which='both', direction='in', top=True, right=True, )
            ax[1,1].tick_params(axis='both', which='both', direction='in', top=True, right=True, labelleft=False)
            ax[0,1].tick_params(axis='x', which='minor', bottom=False, top=False)
            ax[1,1].tick_params(axis='x', which='minor', bottom=False, top=False)
            
            if key=='pt_jet': 
                ax[0,0].legend(frameon=False, fontsize='small', loc=1)
            if key=='kt_DyG':
                ax[0,0].text(.1,.96, r'hardest jet in pp@5.02TeV', fontsize='small', ha='left', va='top', transform=ax[0,0].transAxes)
                ax[0,0].text(.1,.89, r'$p_T>250$ GeV, $R_{ak_t}=0.4, |\eta|<2.8$', fontsize='small', ha='left', va='top', transform=ax[0,0].transAxes)
            if key=='tau2': 
                ax[0,1].text(2.5,1, f'Truth/Measured: {nature_label}', color='gray', fontsize='small', ha='right', va='top', transform=ax[0,1].transAxes, rotation=-90)
                ax[0,1].text(2.1,1, f'Generated/Simulated: {synthetic_label}', color='gray', fontsize='small', ha='right', va='top', transform=ax[0,1].transAxes, rotation=-90)

            pdf.savefig(bbox_inches='tight')  
            plt.close()
print('Done.')