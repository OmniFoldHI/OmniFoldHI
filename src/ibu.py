import numpy as np
import copy

def MChistogramdd(data, bins, w=None, density=False, probability=False, dividebin=True):
    ''' Histogram of N-dim data: dN / dx1*dx2*...*dxn. 
        It also returns with uncertainty assuming the normal distributed bincounts 
        typical for MC generated smaples.
       
        Parameters
        ----------
        data : [data1, data2, ..., dataN], list of array input data.
        bins : [bins1, bins2, ..., binsN], containing bin edges.
        w    : array, weight of each entry.
        density     : bool, normalizing the histogram to 1.
        probability : bool, dividing by the number of events.
        dividebin   : bool, dividing by the binwidth.
              
        Returns
        -------
        [z, zerr, bins]
    '''
    if len(data) != len(bins): 
        return print('Error data and bins are incompatible', len(data), len(bins))
    
    if w is None: w = np.ones(len(data[0]))

    z    = np.histogramdd(data, bins=bins, weights=w,    density=False)[0]
    zerr = np.histogramdd(data, bins=bins, weights=w**2, density=False)[0]
    
    z_sum = 1
    if density:     z_sum = z.sum()
    if probability: z_sum = w.sum()
    
    if dividebin:
        bindiff = [np.diff(i) for i in bins]
        for i in range(len(data)):
            shape    = np.ones(len(bins), int)
            shape[i] = len(bins[i]) - 1
            z    = z / bindiff[i].reshape(shape)
            zerr = np.sqrt(zerr) / bindiff[i].reshape(shape)
    z    /= z_sum
    zerr /= z_sum
    
    return [z, zerr, bins]

def response(evs, bins):
    ''' Response matrix, normalized to be probability including Fakes and Trash 
        in the zeroth bins: j = [0, 1, ..., nE], i = [0, 1, ..., nC]
        Algorithm
        ---------
        lam[j1, j2, ..., i1, i2, ...] = P(Ej1, Ej2, ...| Ci1, Ci2, ...) 
                                      = P(Ej1, Ej2, ..., Ci1, Ci2, ...) / P(Ci1, Ci2, ...).
        
        Parameters
        ----------
        evs_e = [arr_e1, arr_e2, ...] : list of array of causes.
        evs_c = [arr_c1, arr_c2, ...] : list of array of effects.
        bins  = [bins_e, bins_c] : list of array of bins including Trash and Fake.
        Returns
        -------
        lam = [p_ec, p_ec_err, bins]
    '''
    evs_e,  evs_c  = evs
    bins_e, bins_c = bins
    
    # Joint probability P(Ej1, Ej2, ..., Ci1, Ci2, ...) and its stat. unc.
    lam, err = MChistogramdd(evs_e+evs_c, bins=bins_e+bins_c, probability=True)[0:2]
    
    # Conditioned probability: 
    # P(Ej1, Ej2, ...| Ci1, Ci2, ...) = P(Ej1, Ej2, ..., Ci1, Ci2, ...) / P(Ci1, Ci2, ...).
    H_c = MChistogramdd(evs_c, bins=bins[1], probability=True)[0]
    # Expand dimensions of P(Ci1, Ci2, ...) --> P(None, None, ..., Ci1, Ci2, ...) 
    # to evaluate the ratio.
    exp_shape = [None for j in range(len(bins_e))]+[slice(None) for i in range(len(bins_c))]
    lam /= H_c[tuple(exp_shape)] + 1e-50
    err /= H_c[tuple(exp_shape)] + 1e-50
    
    return [lam, err, bins]

def eps(p_ec_mc):
    ''' Calculating the efficiency factor. The measurement doesn't know about Trash, 
        and measured histograms are not probabilities but self normalized densities. 
        Efficiency corrects for this. Here is a derivation of what do we calculate 
        as in higher dimension it is quite complicated.
        Trash and Fake are in j=0 and i=0.
        
        xC[i1,i2,...] 
        = sum_{j1,j2,...=0}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...]
        = sum_{j1,j2,...=1}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...]
          + th[i1,...,0,0,...] * xE[0,0,...] * dxE[0,0,...]
          + Permut sum_{jk=1}^nE      th[i1,..., 0,...,jk,...,0]           * xE[0,...,jk,...,0]          * dxE[0,...,jk,...,0]
          + Permut sum_{jk1,jk2=1}^nE th[i1,..., 0,...,jk1,...,jk2,...,0]  * xE[0,...,jk1,...,jk2,...,0] * dxE[0,...,jk1,...,jk2,...,0]
          + ...
          + Permur sum_{j1,...,NOT jk,...=1}^nE th[i1,...,j1,...,jk=0,...] * xE[j1,...,jk=0,...]         * dxE[j1,...,jk=0,...]
        
        Using Bayes theorem, invert th-->lam for terms containing Trash:
        = sum_{j1,j2,...=1}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...]
          + ( lam[0,..., i1,...] * dxE[0,...,jk,...,0] 
            + Permut sum_{jk=1}^nE                lam[0,...jk,...,0, i1,...]   * dxE[0,...,jk,...,0]
            + ...
            + Permut sum_{j1,...,NOT jk,...=1}^nE lam[j1,...,jk=0,..., i1,...] * dxE[j1,...,jk=0,...] 
            ) * xC[i1,i2,...]
        = sum_{j1,j2,...=1}^nE th[i1,i2,..., j1,j2,...] * xE[j1,j2,...] * dxE[j1,j2,...] / (1 - eps[i1,i2,...])
        
        This function returns with 
        eps[i1,i2,...]
        = ( lam[0,..., i1,...] * dxE[j1,j2,...]
          + Permut sum_{jk=1}^nE                lam[0,...jk,...,0, i1,...]   * dxE[0,...,jk,...,0]
          + ...
          + Permut sum_{j1,...,NOT jk,...=1}^nE lam[j1,...,jk=0,..., i1,...] * dxE[j1,...,jk=0,...]  
          ).
                        
        We extend eps[i1,i2,..., j1,j2,...] making it easier to take the ratio.
    '''
    # Binwidth
    dbin_e = [np.diff(i) for i in p_ec_mc[-1][0]]
    dbin_c = [np.diff(i) for i in p_ec_mc[-1][1]]
    for i in range(len(dbin_e)):
        # Create a tuple for the binwidth in a shape of [j1,j2,...].
        if i==0: dbin_e_outer = dbin_e[i]
        else: dbin_e_outer = np.expand_dims(dbin_e_outer, axis=-1) * dbin_e[i]
    
    # The efficiency definition can be viewed as a sum of lam elements where 
    # at least one of the j index is 0 (Trash value).
    # All indices which are 0 at least once.
    mask = (np.indices(dbin_e_outer.shape)==0).any(axis=0)
    res = (np.expand_dims(dbin_e_outer[mask], axis=tuple(-np.arange(1,len(dbin_c)+1))) * p_ec_mc[0][mask]).sum(axis=0)
    # Extending the dimensions eps[i1,i2] --> eps[i1,i2,...,j1,j2,...].
    res = np.expand_dims(res, axis=tuple(-np.arange(1,len(dbin_e)+1)))
    
    return res


def ibu (evs_e_mc, evs_c_mc, bins_mc, val_trash=-1, val_fake=-1,
         evs_e=None, bins_e=None, out_probability=False,
         ibu_itnum=5,
         measure=None, prior=None,
         wp=None, wm=None):
    ''' Iterative Bayesian Unfolding in multi dimensions (D'Agostini 95)
        Parameters
        ----------
        evs_e_mc  = [arr_evs_e1, arr_evs_e2, ...] : list of mc effects.
        evs_c_mc  = [arr_evs_c1, arr_evs_c2, ...] : list of mc causes.
        bins_mc   = [bins_e_mc, bins_e_mc]  : list of mc bins.
        val_trash = -1 : value of fake in causes .
        val_fake  = -1 : value of trash in effects.
        evs_e     = [arr_evs_e1, arr_evs_e2, ...] : list of effects.
        bins_e    = [bins_e1, bins_e2, ...] : list of effect bins.
        out_probability = False : posterior is bincount or probability.
        ibu_itnum = 5 : number of ibu iterations.
        
        Functions
        ---------
        lam(evs_e, evs_c, bins), response matrix P(E| C).
        ibu(measure, prior), ibu algorighm.
        ibu_err(prior_err, measure_err, prior_err_itnum, measure_err_itnum), ibu aldorithm with uncertainty.
        
        
        Iterative Bayesian Unfolding in multi dimensions (D'Adostini 95').
        It includes additional notion of Trash and Fakes.
        Algorithm
        ---------
        x(Ci1, Ci2, ...) 
        = sum_{j1, j2, ...=0}^nE P(Ci1, Ci2, ... | Ej1, Ej2, ...) * x(Ej1, Ej2, ...), 

        where ,

        P(Ci1, Ci2, ... | Ej1, Ej2, ..) 
        = P(Ej1, Ej2, ... | Ci1, Ci2, ...) * P(Ci1, Ci2, ...) / P(Ej1, Ej2, ...).

        If x(Ej) doesn't include the Trash bin, the efficiency is considered.
        
        Parameters
        ----------
        meas  : x(Ej) = [hist, hist_err, bins] measured histogram. Contain trash bin or not.
        prior : P(Ci) = [hist, hist_err, bins] prior histogram.

        Returns
        -------
        pos : x(Ci) = [hist, bins] posterior distribution.

        Plotting
        --------
        plt.plot(bins[:-1]+np.diff(bins)/2, hist)
    '''
    
    # Trash and Fake bins will be placed below the user bins. 
    # Check if the values are below the bin ranges.
    if ( any([val_trash > min(j) for j in bins_mc[0]]) or 
         any([val_fake  > min(i) for i in bins_mc[1]]) ): 
        return print('Error in Trash or Fake bin value.')
    # Append the MC bins with the Trash and Fake bins.
    bins_mc_new = [[np.append(val_trash, j) for j in bins_mc[0]],
                   [np.append(val_fake,  i) for i in bins_mc[1]]]
    
    p_e_mc  = MChistogramdd(evs_e_mc, bins=bins_mc_new[0], probability=True) # MC effect P(Ej | MC) with Trash.
    p_c_mc  = MChistogramdd(evs_c_mc, bins=bins_mc_new[1], w=wp, probability=True) # MC cause P(Ci | MC) with Fake.
    p_ec_mc = response([evs_e_mc, evs_c_mc], bins_mc_new) # MC response P(Ej | Ci, MC) with Trash and Fake.
            
    # Measurement input
    if evs_e != None:
        x_e = MChistogramdd(evs_e, bins=bins_e, w=wm)

    # Prior: P(Ci1, Ci2, ...).
    # It is arbitrary and sources uncertainty. More the 
    # iteration, less the dependence on the prior.
    if prior != None: 
        # User defined prior.
        pri, pri_err, bins_c_mc = prior
    else:
        # Method 1. assuming the MC cause as a prior.
        pri, pri_err, bins_c_mc = p_c_mc
        ## TODO: Method 2. assuming flat cause as a prior.
        ##pri = np.ones(pri.shape)
    shape_c_mc = np.array(pri.shape)
    shape_e_mc = np.array(p_e_mc[0].shape)
        
    # Measured histogram: x(Ej1, Ej2, ...).
    # It is given and sources statistical uncertainty.
    if measure != None:
        meas, meas_err, bins_e = measure
    else:
        meas, meas_err, bins_e = x_e
    shape_e = np.array(meas.shape)

    # Bin widths are needed for summation in various shapes.            
    dbin_c_mc = [np.diff(i) for i in bins_c_mc]
    for i in range(len(dbin_c_mc)):
        if i==0: dbin_c_outer = dbin_c_mc[i]
        else: dbin_c_outer = np.expand_dims(dbin_c_outer, axis=-1) * dbin_c_mc[i]
    dbin_e = [np.diff(i) for i in bins_e]
    for i in range(len(dbin_e)):
        if i==0: dbin_e_outer = dbin_e[i]
        else: dbin_e_outer = np.expand_dims(dbin_e_outer, axis=-1) * dbin_e[i]

    # The posterior and response matrix are flattened. 
    pos = [pri.flatten()]
    lam = p_ec_mc[0].reshape(np.prod(shape_e_mc), np.prod(shape_c_mc))
    for it in range(ibu_itnum):
        # Iteration loop of the unfolding.
        # Inverted MC response with the iterated prior: 
        # th[i1, i2, ..., j1, j2, ...] = P(Ci1, Ci2, ...| Ej1, Ej2, ...) 
        #                              = P(Ej1, Ej2, ...| Ci1, Ci2, ...) * P(Ci1, Ci2, ...) / P(Ej1, Ej2, ...),
        # where P(Ej1, Ej2, ...) = sum_{i1,i2, ...=0}^nC P(Ej1, Ej2, ...| Ci1, Ci2, ...) * P(Ci1, Ci2, ...).
        # Here we flatten the arrays so: th[i12..., j12...].
        th  = lam * pos[-1][None,:]
        # When summing Ci, bin width is considered.
        th /= (th * dbin_c_outer.flatten()[None,:]).sum(axis=1)[:,None] + 1e-50
        th  = th.T
            
        if th.shape[1] != meas.flatten().shape[0]: 
            # Efficiency is handled if there is no Trash bin in the measurement.
            # x(Ci) = sum_j P(Ci | Ej) * x(Ej) + P(Ci | T) * x(T)
            #       = sum_j P(Ci | Ej) * x(Ej) + P(T | Ci) * x(Ci)
            #       = sum_j P(Ci | Ej) * x(Ej) / (1 - P(T | Ci))
            # We include the efficiency factor in th[i1,i2,...,j1,j2,...].
            th  = th.reshape(np.append(shape_c_mc, shape_e_mc))
            #eps = eps(p_ec_mc)
            th  = th / (1. - eps(p_ec_mc) + 1e-50)
            # Trimming the Trash bin of th_ij = th[i1=:, i2=:, ..., j1=1:, j2=1:, ...].
            new_shape = [slice(None) for i in range(th.ndim//2)] + [slice(1,None) for i in range(th.ndim//2)]
            th = th[tuple(new_shape)]
            # Flatten th[i1,i2,...,j1,j2,...] --> th[i12..., j12...].
            th = th.reshape(np.prod(shape_c_mc), np.prod(shape_e))
            
        # Posterior.
        # x(Ci1, Ci2, ...) = sum_{j1,j2, ...} P(Ci1, Ci2, ...| Ej1, Ej2, ...) * x(Ej1, Ej2, ...)
        # When summing Ej, bin width is considered.
        xC = (th * meas.flatten()[None,:] 
                * dbin_e_outer.flatten()[None,:]).sum(axis=1)
        
        if out_probability: 
            # Normalize the posterior to probability if needed.
            # Note, that sum_i xC = 1 including the Fake!
            xC /= sum(xC * dbin_c_outer.flatten()) 
            
        pos.append(xC)
        ## TODO: Early stopping is when the posterior barely changes.
        #if abs(pos[-1]-pos[-2]).sum()/(pos[-1]-pos[-2]).std()<1e-5: break
            
    return [pos[-1].reshape(shape_c_mc), bins_c_mc]

def ibu_err(evs_e_mc, evs_c_mc, bins_mc, val_trash=-1, val_fake=-1,
            evs_e=None, bins_e=None, out_probability=False,
            ibu_itnum=5,
            measure=None, prior=None,
            prior_err_itnum=3, meas_err_itnum=3):
    ''' Estimating uncertainties for ibu.
        
        Algorithm
        ---------
        P(Ci | MC) each event assumed to have Poisson distribution.
        
        Parameters
        ----------
        prior_err_itnum : number of iteration for prior statistical uncertainty.
        meas_err_itnum  : number of iteration for measure statistical uncertainty.
        Returns
        -------
        !!! NOT UP TO DATE !!!
        pos : x(Ci) = [hist, hist_err, bins] posterior distribution.
        Plotting
        --------
        plt.errorbar(bins[:-1]+np.diff(bins)/2, pos_new_err[0], yerr=pos[1], xerr=np.diff(bins)/2)
    '''
    pos = ibu(evs_e_mc, evs_c_mc, bins_mc, val_trash, val_fake,
              evs_e, bins_e, out_probability,
              ibu_itnum,
              measure, prior)
    bins = pos[-1]
    pos_err = [ pos[0] ]
        
    for i in range(prior_err_itnum):
        # Prior uncertainty.
        # Sampling several equally possible prior probability.
        # TODO: the sampling could be done on the histogram itself,
        # speeding up the algorithm.
        wp  = np.random.poisson(1, size=np.asarray(evs_c_mc).shape[1])
        for j in range(meas_err_itnum):
            # Measurememnt uncertainty.
            # Sampling several equally possible measurements.
            wm   = np.random.poisson(1, size=np.asarray(evs_e).shape[1])
            pos  = ibu(evs_e_mc, evs_c_mc, bins_mc, val_trash, val_fake,
                       evs_e, bins_e, out_probability,
                       ibu_itnum,
                       measure, prior,
                       wp, wm)
            pos_err.append(pos[0])
                            
    # Calculating the mean and the standard deviation.
    pos_mean = np.mean(np.asarray(pos_err), axis=0)
    pos_var  = np.std(np.asarray(pos_err), axis=0)

    posdd_err = [pos_mean, pos_var, bins]

    hist_mean = []
    hist_err  = []

    if len(bins) == 1:

        hist = np.vstack(( posdd_err[0] , posdd_err[1] )).T
        hist[0] *= np.diff(posdd_err[-1][0])[0]
        hist_mean.append( hist.T[0] )
        hist_err.append( hist.T[1] )

    else:
        # Multi-dim bin volumes.
        dbin_c = [np.diff(j) for j in posdd_err[-1]]
        #print(len(dbin_c)-1)
        for i in range(len(dbin_c)-1):
            if i==0: dbin_c_outer = dbin_c[i]
            dbin_c_outer = np.expand_dims(dbin_c_outer, axis=-1) * dbin_c[i+1]

        for i in range(len(bins)):
            
            axis = tuple([j for j in range(len(posdd_err[-1])) if j!=i])

            hist = np.vstack(( (posdd_err[0] * dbin_c_outer).sum(axis=axis)/dbin_c[i] , (posdd_err[1] * dbin_c_outer).sum(axis=axis)/dbin_c[i] )).T
            hist[0] *= np.diff(posdd_err[-1][i])[0]
            hist_mean.append( hist.T[0] )
            hist_err.append( hist.T[1] )

    return [hist_mean, hist_err, bins]


###############################
### N E W   C O D E ###########
###############################

def ibu_from_dict (dict_data, obs_unfold, n_iterations, prior_err_itnum, meas_err_itnum, retrun_dict_ibu=True):

    evs_list_syn_E = [ dict_data[obs_label]['evs_syn_E'] for obs_label in obs_unfold ]
    evs_list_syn_C = [ dict_data[obs_label]['evs_syn_C'] for obs_label in obs_unfold ]
    evs_list_nat_E = [ dict_data[obs_label]['evs_nat_E'] for obs_label in obs_unfold ]  # TAKE OUT TRASH FROM HERE!!!! or make sure it is not there!

    bins_list_C  = [ dict_data[obs_label]['bins_C'] for obs_label in obs_unfold ]
    bins_list_E  = [ dict_data[obs_label]['bins_E'] for obs_label in obs_unfold ]
    bins_list_CE = [bins_list_C, bins_list_E]

    val_trash = dict_data[obs_unfold[0]]['val_T']
    val_fake  = dict_data[obs_unfold[0]]['val_F']

    # create dictionary with unfolded info and histograms
    dict_ibu = {}

    # copy relevant info from dict_data to dict_ibu, so dict_ibu can bve used alone
    keys_to_copy = ['xlim_C', 'xlim_E', 'xlabel', 'ylabel', 'val_F', 'val_T',
                    'bins_C', 'bins_E', 'midbin_C', 'midbin_E', 'midbin_C_unc', 'midbin_E_unc']
    for obs_label in obs_unfold:
        dict_ibu[obs_label] = { key: copy.deepcopy(dict_data[obs_label][key]) for key in keys_to_copy }

    [hist_mean, hist_err, bins] = ibu_err(evs_list_syn_E, evs_list_syn_C, bins_list_CE, val_trash=val_trash, val_fake=val_fake,
                                          evs_e=evs_list_nat_E, bins_e=bins_list_E, out_probability=True,
                                          ibu_itnum=n_iterations,
                                          measure=None, prior=None,
                                          prior_err_itnum=prior_err_itnum, meas_err_itnum=meas_err_itnum)

    # new return to be consistent w/ omnifold
    histogram_list_data = []
    for obs_label, hist_C, hist_C_unc, bins_C in zip(obs_unfold, hist_mean, hist_err, bins):

        # bins
        bins_C = bins_C[1:]
        binwidth_C   = bins_C[1] - bins_C[0]  # assuming linear binning
        midbin_C     = (bins_C[:-1]+bins_C[1:])/2
        midbin_C_unc = binwidth_C/2 * np.ones(len(midbin_C))

        # bins to return
        midbin_FTC        = np.hstack([ val_fake , val_trash , midbin_C ])
        midbin_FTC_unc    = np.hstack([ 0        , 0         , midbin_C_unc ])

        # histograms to return
        hist_FTC     = np.hstack([ hist_C[0]    , np.nan, hist_C[1:] ])
        hist_FTC_unc = np.hstack([ hist_C_unc[0], np.nan, hist_C_unc[1:] ])

        # append to observables' list of histograms
        histogram_list_data.append( np.vstack([ midbin_FTC, midbin_FTC_unc, hist_FTC, hist_FTC_unc ]) )

        # add histos to dict_ibu
        dict_ibu[obs_label]['hist_unf_C'] = hist_FTC[2:]
        dict_ibu[obs_label]['hist_unf_C_unc'] = hist_FTC_unc[2:]

        dict_ibu[obs_label]['hist_unf_F'] = hist_FTC[0]
        dict_ibu[obs_label]['hist_unf_F_unc'] = hist_FTC_unc[0]

        dict_ibu[obs_label]['hist_unf_T'] = dict_data[obs_label]['hist_syn_T'][0]
        dict_ibu[obs_label]['hist_unf_T_unc'] = dict_data[obs_label]['hist_syn_T_unc'][0]

    if retrun_dict_ibu:
        return dict_ibu
    else:
        return histograms_data

    















        
            



































#def MChistogramdd(data, bins, weights=None, density=False, probability=False, dividebin=True):
#    '''
#    Histogram of N-dim data: dN / dx1*dx2*...*dxn. 
#    It also returns with uncertainty assuming the normal distributed bincounts 
#    typical for MC generated smaples.
#    
#    Parameters
#    ----------
#    data : [data1, data2, ..., dataN], list of array input data.
#    bins : [bins1, bins2, ..., binsN], containing bin edges.
#    w    : array, weight of each entry.
#    density     : bool, normalizing the histogram to 1.
#    probability : bool, dividing by the number of events.
#    dividebin   : bool, dividing by the binwidth.
#            
#    Returns
#    -------
#    [z, z_err, bins]
#    '''
#
#    if len(data) != len(bins): 
#        raise Exception('Error data and bins are incompatible', len(data), len(bins))
#
#    if not weights:
#        weights = np.ones(len(data[0]))
#
#    # Compute histograms
#    z     = np.histogramdd(data, bins=bins, weights=weights   , density=False)[0]
#    z_err = np.histogramdd(data, bins=bins, weights=weights**2, density=False)[0] **.5
#    
#    z_sum = 1
#    if density:     z_sum = z.sum()
#    if probability: z_sum = w.sum()
#    
#    if dividebin:
#        bindiff = [np.diff(i) for i in bins]
#        for i in range(len(data)):
#            shape    = np.ones(len(bins), int)
#            shape[i] = len(bins[i]) - 1
#            z    = z / bindiff[i].reshape(shape)
#            z_err = np.sqrt(z_err) / bindiff[i].reshape(shape)
#    z     /= z_sum
#    z_err /= z_sum
#    
#    return [z, z_err, bins]
#
#
#def response_matrix (events_cause, events_effect, bins_cause, bins_effect):
#    '''
#    Response matrix, normalized to be probability including Fakes and Trash 
#    in the zeroth bins: j = [0, 1, ..., nE], i = [0, 1, ..., nC]
#    Algorithm
#    ---------
#    lam[j1, j2, ..., i1, i2, ...] = P(Ej1, Ej2, ...| Ci1, Ci2, ...) 
#                                    = P(Ej1, Ej2, ..., Ci1, Ci2, ...) / P(Ci1, Ci2, ...).
#    
#    Parameters
#    ----------
#    evs_e = [arr_e1, arr_e2, ...] : list of array of causes.
#    evs_c = [arr_c1, arr_c2, ...] : list of array of effects.
#    bins  = [bins_e, bins_c] : list of array of bins including Trash and Fake.
#    Returns
#    -------
#    lam = [p_ec, p_ec_err, bins]
#    '''
#    
#    # Joint probability P(Ej1, Ej2, ..., Ci1, Ci2, ...) and its stat. unc.
#    lam, err = MChistogramdd(events_effect+events_cause, bins=bins_effect+bins_c, probability=True)[0:2]
#    
#    # Conditioned probability: 
#    # P(Ej1, Ej2, ...| Ci1, Ci2, ...) = P(Ej1, Ej2, ..., Ci1, Ci2, ...) / P(Ci1, Ci2, ...).
#    H_c = MChistogramdd(events_cause, bins=bins_cause, probability=True)[0]
#    # Expand dimensions of P(Ci1, Ci2, ...) --> P(None, None, ..., Ci1, Ci2, ...) 
#    # to evaluate the ratio.
#    exp_shape = [None for j in range(len(bins_effect))]+[slice(None) for i in range(len(bins_cause))]
#    lam /= H_c[tuple(exp_shape)] + 1e-50
#    err /= H_c[tuple(exp_shape)] + 1e-50
#    
#    return [lam, err, bins]
#
#
#
#
#
#def ibu (events_syn_cause, events_syn_effect, events_nat_effect, bins_cause, bins_effect,
#         val_trash=-1, val_fake=-1):
#    '''
#    Iterative Bayesian Unfolding (IBU)
#
#    Parameters
#    ----------
#    events_syn_cause : array
#    events_syn_effect : array
#    events_nat_effect : array
#    bins_cause : array
#    bins_effect : array
#    val_trash : float
#    val_fake : float
#
#    Returns
#    -------
#    . : unfolded histogram
#
#    '''
#
#    # Check that val_trash or val_fake not inside or above bins
#    if any( val_trash >= bins_effect ) or any( val_fake >= bins_cause ):
#        raise Exception('Error in trash or fake bin value.')
#
#    # Create trash and fake bins
#    bins_new_cause  = np.append(val_fake , bins_cause)
#    bins_new_effect = np.append(val_trash, bins_effect)
#
#    # Compute P(Ej|MC), P(Ci|MC) and response matrix P(Ej|Ci,MC) w/ trash and fakes
#    prob_effect_syn = MChistogramdd(events_syn_effect, bins=bins_new_effect, probability=True)
#    prob_cause_syn  = MChistogramdd(events_syn_cause , bins=bins_new_cause , probability=True)
#    response_syn    = response_matrix()
#
#
#
#    p_ec_syn = lam([self.evs_e_mc, self.evs_c_mc], self.bins_mc_new)
#
#
#
#
#
#
#
#    return bins_new_cause
#