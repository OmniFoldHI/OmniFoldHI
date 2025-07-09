import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import json
import matplotlib.colors
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

def response(evs_e, evs_c, bins, val_trash, val_fake):
    ''' Response matrix, normalized to being probability including fakes and trash.
        Algorithm
        ---------
        lam[j,i] = P(Ej | Ci) = P(Ci, Ej) / P(Ci).
    '''
    # Introduce trash and fake bins as new 0th bins.
    bins_new = [np.append(val_trash, bins[0]), np.append(val_fake, bins[1])]
    # Constructing lam using the joint probability.
    lam = MChistogram2d(evs_e, evs_c, bins=bins_new, probability=True)
    H_c = MChistogram(evs_c, bins=bins_new[1], probability=True)
    lam[0] /= H_c[0][None,:] + 1e-50
    lam[1] /= H_c[0][None,:] + 1e-50
    return lam

def MChistogram2d(datax, datay, bins, w=None, density=False, probability=False, dividebin=True):
    ''' Histogram of 2d data: dN/dxdy. 
        It also returns with uncertainty assuming the normal distributed 
        bincounts (typical for MC generated smaples).
       
        Parameters
        ----------
        datax : array, input data1.
        datay : array, input data2.
        bins  : (array, array), containing x and y bin edges.
        w     : array, weight of each entry.
        density     : bool, normalizing the histogram to 1.
        probability : bool, dividing by the number of events.
        dividebin   : bool, dividing by the binsize.
              
        Returns
        -------
        [z, zerr, xbins, ybins]
        
        Use pcolormesh(X,Y,Z) for plotting.
    '''
    if w is None: w = np.ones(len(datax))
    z, x, y        = np.histogram2d(datax, datay, bins=bins, weights=w,    density=False)
    zerr, dum, dum = np.histogram2d(datax, datay, bins=bins, weights=w**2, density=False)
    if density==True:
        return [z/np.outer(np.diff(x),np.diff(y))/sum(z), np.sqrt(zerr)/np.outer(np.diff(x),np.diff(y))/sum(z), x, y]
    if probability==True:
        return [z/np.outer(np.diff(x),np.diff(y))/sum(w), np.sqrt(zerr)/np.outer(np.diff(x),np.diff(y))/sum(w), x, y]
    if dividebin==True:
        return [z/np.outer(np.diff(x),np.diff(y)),        np.sqrt(zerr)/np.outer(np.diff(x),np.diff(y)),        x, y]
    return [z, np.sqrt(zerr), x, y]

# Load dataset
print('Load data.')
# p8_c = np.loadtxt("../../data/Pythia_bkg0_Delphes_Generated.out")
# p8_e = np.loadtxt("../../data/Pythia_bkg1_Delphes_Simulated.out")
h7_c = np.loadtxt("../../data/Herwig_bkg0_Delphes_Generated.out")
h7_e = np.loadtxt("../../data/Herwig_bkg1_Delphes_Simulated.out")

# Load observable format
with open('../config_dict_data.json') as f:
    obs = json.load(f)

nature_label = 'Herwig7 (+ Bkg. + Delphes)'
synthetic_label = 'Pythia8 (+ Bkg. + Delphes)'

print('Plotting. (might take a few minutes)')
with mpl.rc_context({'font.family': 'sans-serif', 'font.size': 14, 'text.usetex':False}):

    hist_style = {
        'tru': {'label':'Truth', 'lw':2, 'color':'green'},
        'dat': {'label':'dat', 'lw':2, 'color':'k'}
    }

    with PdfPages('../../figures/response.pdf') as pdf:

        for i,ikey in enumerate(obs):
            for j,jkey in enumerate(obs):
                print(ikey,jkey)

                # Data out of binrange is trash
                effect = np.array([val if (val!=-7 and obs[ikey]['xlim_E'][0]<=val<=obs[ikey]['xlim_E'][1]) else -7 for val in h7_e.T[i]])
                cause  = np.array([val if (val!=-7 and obs[jkey]['xlim_C'][0]<=val<=obs[jkey]['xlim_C'][1]) else -7 for val in h7_c.T[j]])

                xbins_e = np.linspace(*obs[ikey]['xlim_E'], obs[ikey]['nbins_E'])
                xbins_c = np.linspace(*obs[jkey]['xlim_C'], obs[jkey]['nbins_C'])
                h1 = response(effect, cause, [xbins_e, xbins_c], -7, -7)
                h1[0] *= np.diff(h1[-2])[:,None]

                # Create figure and grid
                fig, ax = plt.subplots(2,2, figsize=(8,7), height_ratios=(13,1), width_ratios=(1,15), gridspec_kw={'hspace': 0.03, 'wspace': 0.03})

                X, Y = np.meshgrid(h1[-1][:,np.newaxis], h1[-2])
                ax[0,1].pcolormesh(X.T[1:,1:], Y.T[1:,1:], h1[0].T[1:,1:], norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=1))
                X0 = np.array([np.zeros(obs[ikey]['nbins_C']+1),np.ones(obs[ikey]['nbins_C']+1)])
                im = ax[0,0].pcolormesh(X0, Y.T[:2,:], h1[0].T[:1,:], norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=1))
                Y0 = np.array([np.zeros(obs[jkey]['nbins_E']+1),np.ones(obs[jkey]['nbins_E']+1)]).T
                ax[1,1].pcolormesh(X.T[:,:2], Y0, h1[0].T[:,:1], norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=1))
                ax[1,0].pcolormesh([[0,0],[1,1]], [[0,1],[0,1]], h1[0].T[:1,:1], norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=1))

                ax[1,0].set_xticks(ticks=[0.5], labels=[r'$F$'])
                ax[1,0].set_yticks(ticks=[0.5], labels=[r'$T$'])
                ax[0,0].minorticks_on()
                ax[0,1].minorticks_on()
                ax[1,0].minorticks_on()
                ax[1,1].minorticks_on()
                ax[0,0].tick_params(axis='both',which='both', right=True, top=True, bottom=True, labelbottom=False, direction='in', labelsize=20)
                ax[1,0].tick_params(axis='both',which='both', right=True, top=True, bottom=True, direction='in', labelsize=20)
                ax[0,1].tick_params(axis='both',which='both', right=True, top=True, bottom=True, labelleft=False, labelbottom=False, direction='in', labelsize=20)
                ax[1,1].tick_params(axis='both',which='both', right=True, top=True, bottom=True, labelleft=False, direction='in', labelsize=20)
                ax[0,0].set_xlim(0,1)
                ax[0,0].set_ylim(obs[ikey]['xlim_E'])
                ax[1,0].set_ylim(0,1)
                ax[1,0].set_xlim(0,1)
                ax[0,1].set_xlim(obs[jkey]['xlim_C'])
                ax[0,1].set_ylim(obs[ikey]['xlim_E'])
                ax[1,1].set_xlim(obs[jkey]['xlim_C'])
                ax[1,1].set_ylim(0,1)
                ax[0,0].set_ylabel('Measured: '+ obs[ikey]['xlabel'], fontsize=20)
                ax[1,1].set_xlabel('Truth: '+obs[jkey]['xlabel'], fontsize=20)
                cb = fig.colorbar(im, ax=ax.ravel().tolist(), pad=0.01, aspect=50)
                cb.set_label(r'$P(E_j|C_i)$', fontsize=20)
                cb.ax.tick_params(which='both', right=True, top=True, bottom=True, labelleft=False, direction='in', labelsize=20)

                # if ikey=='kt_DyG' and jkey=='tau2':
                #     ax[0,0].text(1.42, 1, r'hardest jet in pp@5.02TeV', color='gray', ha='right', va='top', transform=ax[0,1].transAxes, rotation=-90)
                #     ax[0,0].text(1.35, 1, r'$p_T>250$ GeV, $R_{ak_t}=0.4, |\eta|<2.8$', color='gray', ha='right', va='top', transform=ax[0,1].transAxes, rotation=-90)
                # if ikey=='tau2' and jkey=='tau2': 
                #     ax[0,1].text(1.35,1, f'Truth/Measured: {nature_label}', color='gray', ha='right', va='top', transform=ax[0,1].transAxes, rotation=-90)

                pdf.savefig(bbox_inches='tight')  
                plt.close()
print('Done.')