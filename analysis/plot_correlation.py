import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from scipy.cluster import hierarchy
from scipy import stats
from matplotlib.colors import to_hex

# Observable names
obs_names = [r'$p_T$',r'$\eta$',r'$\phi$',r'$m$',r'$n$',
             r'$z_g$',r'$\theta_g$',r'$k_{T,g}$',
             r'$p_T^{sd}$',r'$m_{sd}$',r'$z_{sd}$',r'$p_T^{rsd}$',r'$n_{rsd}$',
             r'$\tau_1$',r'$\tau_2$',r'$\tau_3$',r'$\tau_4$',r'$\tau_5$']

# Correlation matrix (Pearson's coefficient):
# print('Calculating the correlations might take a few minutes. Consider loading ' \
#     'the npy file instead of recalculating it.')
# # Load data
# cause  = np.loadtxt("../../data/Herwig_bkg0_Delphes_Generated.out")
# # effect = np.loadtxt("../../data/Herwig_bkg1_Delphes_Simulated.out")
# effect = cause
# Corr = []
# for i,e in enumerate(effect.T):
#     print(i, '/', len(obs_names))
#     for c in cause.T:
#         # Strip fakes and trash for simplicity.
#         temp = np.array([i for i in np.array([e, c]).T if all(0<i) and all(i==i)])
#         Corr.append(stats.pearsonr(temp.T[0], temp.T[1])[0])
# Corr = np.array(Corr).reshape(len(obs_names),len(obs_names))
# Corr = np.save('CorrMX', Corr)
Corr = np.load('CorrMX.npy') # Use load for speedup.

# Figure
with mpl.rc_context({'font.family': 'sans-serif', 'font.size': 14, 'text.usetex':False}):

    fig, ax = plt.subplots(2, 1, figsize=(8,10), height_ratios=(7,1), gridspec_kw={'hspace': 0.05, 'wspace': 0.0}) 

    # The hierarchy clustering needs distances. We use dij =  1 - |Cij|.
    # hierarchy.linkage only needs the upper diagonal of the correlation matrix.
    dist = (1-abs(Corr))[np.triu_indices(18, k=1)]
    Z = hierarchy.linkage(dist, method='average', metric='euclidean')

    # Plot clustering sequence.
    n_clusters = [1,5,11,16]
    d_colors   = [1,5,10,15]
    cmap_mfHI = plt.get_cmap('YlOrRd')
    colors = [to_hex(cmap_mfHI((d+5)/(18+5))) for d in d_colors]
    for i,n in enumerate(n_clusters):
        #above_threshold_color = 'grey' if i==0 else 'none'
        above_threshold_color = colors[0] if i==0 else 'none'
        color_threshold = Z[-(n-1),2]
        hierarchy.set_link_color_palette([colors[i]])
        dn = hierarchy.dendrogram(
            Z,
            labels=obs_names,
            orientation='bottom',
            ax=ax[1],
            color_threshold=color_threshold,
            above_threshold_color=above_threshold_color
        )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([])

    # The clustering tree is ordered in dij = 1-|Cij| pair distances. 
    # The correlation matrix follows obs_names ordering. We reorder the corr mx
    # to match the clustering tree. 
    def get_permutation_indices(original, permuted):
        return [original.index(item) for item in permuted]
    new_order = get_permutation_indices(obs_names, dn['ivl'])
    res_new = Corr[np.ix_(new_order, new_order)]

    # Plot correlation matrix
    im = ax[0].matshow(res_new, vmin=-1, vmax=1, cmap='coolwarm')
    cax = fig.add_axes([ax[0].get_position().x1+0.01, ax[0].get_position().y0, 0.03, ax[0].get_position().height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'Linear Correlation')
    ax[0].set_xticks(range(len(obs_names)), labels=dn['ivl'])
    ax[0].set_yticks(range(len(obs_names)), labels=dn['ivl'])
    ax[1].set_xlabel(r'Truth')
    ax[0].set_ylabel(r'Truth')
    ax[0].tick_params(axis='both',which='both', left=False, right=False, top=False, bottom=False, labelbottom=True,labeltop=False, direction='in')

    plt.savefig('../figures/correlation.pdf', bbox_inches='tight')